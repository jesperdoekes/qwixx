#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import itertools, json
from typing import Dict, List, Tuple
from qwixx.autoplay import play_series
from qwixx import ai as ai_mod

OUTDIR = Path("./tournament_heur_test_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- Configure the experiment ----------
SEED_BASE = 13579
GAMES = 300                 # start with 300; confirm winners at 800–1000
RANDOM_PROFILES = 8         # how many random HEUR profiles to spawn
PROFILE_PREFIX = "rand"

# Baselines to test against
BASELINES = ["gap1", "gap1lock", "gap2", "gap2lock"]

# Ranges for random profiles (feel free to widen/narrow)
RANGES = {
    'setup_bonus': (3.8, 6.0),
    'flex_weight': (0.14, 0.26),
    'big_skip_penalty_per_cell': (0.95, 1.30),
    'close_now_bonus': (8.0, 11.0),
    'no_lock_cap_penalty': (5.5, 8.0),
    'min_component_value': (ai_mod.MIN_COMPONENT_VALUE, ai_mod.MIN_COMPONENT_VALUE),
    'combo_delta_margin': (ai_mod.COMBO_DELTA_MARGIN, ai_mod.COMBO_DELTA_MARGIN),
}

def ensure_profiles() -> List[str]:
    """Create a batch of random profiles + a reference 'hand_v1' profile."""
    names = ai_mod.bulk_register_random_profiles(
        count=RANDOM_PROFILES,
        seed=SEED_BASE,
        ranges=RANGES,
        prefix=PROFILE_PREFIX
    )
    # Optional: a hand-tuned baseline near HeuristicStrongBot's defaults
    ai_mod.register_heur_profile("hand_v1", {
        'setup_bonus': 4.5,
        'flex_weight': 0.18,
        'big_skip_penalty_per_cell': 1.10,
        'close_now_bonus': 9.0,
        'no_lock_cap_penalty': 6.0,
        'min_component_value': ai_mod.MIN_COMPONENT_VALUE,
        'combo_delta_margin': ai_mod.COMBO_DELTA_MARGIN,
    })
    (OUTDIR / "profiles.json").write_text(
        json.dumps({k: ai_mod.HEUR_PROFILES[k] for k in names + ["hand_v1"]}, indent=2)
    )
    return names + ["hand_v1"]

def run_pair(kindA: str, kindB: str, order: str, idx: int, games: int) -> Dict:
    """Run a single series and return a row for the results JSON."""
    if order == "AB":
        names = [f"A_{kindA}", f"B_{kindB}"]
        kinds = [kindA, kindB]
    else:
        names = [f"A_{kindB}", f"B_{kindA}"]
        kinds = [kindB, kindA]
    bot_types = {names[0]: kinds[0], names[1]: kinds[1]}
    seed = SEED_BASE + idx
    report = play_series(
        names=names,
        bot_types=bot_types,
        games=games,
        mcts_sims=0, mcts_depth=0,
        seed=seed, show_each=False,
        progress_interval=0.5,
        epsilon=0.10, epsilon_end=0.05, epsilon_decay=0.995,
        learn_file="./qwixx_data/learner.json",
        debug_timing=False
    )
    return {
        "pair_index": idx,
        "kindA": kindA, "kindB": kindB, "order": order,
        "games": report["games"],
        "wins": report["wins"],
        "ties": report["ties"],
        "avg_scores": report["avg_scores"],
        "avg_margin": report["avg_margin"],
    }

def main():
    # 0) Make sure your autoplay factory has:
    #     elif kind.startswith('heur_profile:'):
    #         prof = kind.split(':', 1)[1]
    #         from .ai import HeurProfileBot
    #         return HeurProfileBot(name=name, profile=prof)
    # (Optionally also add 'heur_strong' mapping if you want to include it.)

    # 1) Register random HEUR profiles + hand_tuned
    prof_names = ensure_profiles()
    prof_kinds = [f"heur_profile:{p}" for p in prof_names]

    # 2) Build match list:
    #    - each profile vs baselines (both orders)
    #    - round-robin among profiles (both orders)
    pairs: List[Tuple[str,str]] = []
    for p in prof_kinds:
        for b in BASELINES:
            pairs.append((p, b))
    for i in range(len(prof_kinds)):
        for j in range(i+1, len(prof_kinds)):
            pairs.append((prof_kinds[i], prof_kinds[j]))

    # 3) Run
    results = []
    idx = 0
    for A, B in pairs:
        for order in ("AB", "BA"):
            idx += 1
            print(f"=== Pair {idx}: {A} vs {B} [{order}] ===")
            row = run_pair(A, B, order, idx, GAMES)
            results.append(row)

    # 4) Save results
    out_json = OUTDIR / "heur_test_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out_json}")

    # 5) Print compact Win Rate vs Field (correct A/B mapping)

def accumulate_wins_and_games(results):
    """
    Correctly accumulates wins and games per bot across all series, handling AB/BA.
    Returns: dict bot -> {'wins': int, 'games': int}
    """
    agg = {}
    def bump(bot, w, g):
        a = agg.setdefault(bot, {'wins': 0, 'games': 0})
        a['wins'] += w
        a['games'] += g

    for r in results:
        g = r['games']
        if r['order'] == 'AB':
            a_bot, b_bot = r['kindA'], r['kindB']
            w_a = r['wins'].get(f"A_{a_bot}", 0)
            w_b = r['wins'].get(f"B_{b_bot}", 0)
        else:  # 'BA'
            a_bot, b_bot = r['kindB'], r['kindA']
            w_a = r['wins'].get(f"A_{a_bot}", 0)
            w_b = r['wins'].get(f"B_{b_bot}", 0)

        bump(a_bot, w_a, g)
        bump(b_bot, w_b, g)
    return agg

agg = accumulate_wins_and_games(results)

# Print profiles-only
print("\n=== HEUR profiles: Win Rate vs Field (corrected) ===")
prof_agg = {k:v for k,v in agg.items() if k.startswith("heur_profile:")}
rank = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0, v['wins'], v['games']) for k,v in prof_agg.items()),
              key=lambda t: -t[1])
for k, wr, w, g in rank:
    print(f"{k:>28}  WR={wr*100:5.1f}%  (wins={w}, games={g})")

# Optional: also print all bots
print("\n=== All bots: Win Rate vs Field (corrected) ===")
rank_all = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0, v['wins'], v['games']) for k,v in agg.items()),
                  key=lambda t: -t[1])
for k, wr, w, g in rank_all:
    print(f"{k:>28}  WR={wr*100:5.1f}%  (wins={w}, games={g})")
    
# Sanity: total wins must equal total games - total ties
total_games = sum(r['games'] for r in results)
total_ties  = sum(r.get('ties', 0) for r in results)
wins_sum = 0
for r in results:
    if r['order'] == 'AB':
        wins_sum += r['wins'].get(f"A_{r['kindA']}",0) + r['wins'].get(f"B_{r['kindB']}",0)
    else:
        wins_sum += r['wins'].get(f"A_{r['kindB']}",0) + r['wins'].get(f"B_{r['kindA']}",0)
assert wins_sum == (total_games - total_ties), \
    f"inconsistent: wins={wins_sum} vs games-ties={total_games-total_ties}"

if __name__ == "__main__":
    main()
