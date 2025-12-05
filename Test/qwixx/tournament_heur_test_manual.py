
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
from typing import Dict, List, Tuple

from qwixx.autoplay import play_series
from qwixx import ai as ai_mod

OUTDIR = Path("./tournament_heur_test_out (3)")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------ Configure the experiment ------------------
SEED_BASE = 135800
GAMES = 300  # start with 300; confirm winners at 800–1000

# Baselines: match tournament_heur_G.py exactly
BASELINES = [
    "gap2", "gap2lock",
    "gap3", "gap3lock",
    "gap_adapt3", "gap_adapt3lock",
]

# --- Manually registered HEUR profiles (one-time) ---
# You pasted eight; all eight are included. Comment out two if you truly want six.
MANUAL_PROFILES = {
  "n04-anti2+++": {
    "setup_bonus": 3.40,
    "flex_weight": 0.28,
    "big_skip_penalty_per_cell": 1.35,
    "close_now_bonus": 11.30,
    "no_lock_cap_penalty": 7.90,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "n04-anti2lock+++": {
    "setup_bonus": 3.60,
    "flex_weight": 0.27,
    "big_skip_penalty_per_cell": 1.30,
    "close_now_bonus": 11.10,
    "no_lock_cap_penalty": 7.60,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "n05-anti3+++": {
    "setup_bonus": 5.80,
    "flex_weight": 0.26,
    "big_skip_penalty_per_cell": 1.32,
    "close_now_bonus": 9.30,
    "no_lock_cap_penalty": 5.80,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "n05-anti3lock+++": {
    "setup_bonus": 6.60,
    "flex_weight": 0.26,
    "big_skip_penalty_per_cell": 1.32,
    "close_now_bonus": 8.20,
    "no_lock_cap_penalty": 5.60,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "n06-antiAdapt3+++": {
    "setup_bonus": 5.80,
    "flex_weight": 0.24,
    "big_skip_penalty_per_cell": 1.10,
    "close_now_bonus": 7.80,
    "no_lock_cap_penalty": 5.80,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "n06-antiAdapt3lock+++": {
    "setup_bonus": 4.60,
    "flex_weight": 0.24,
    "big_skip_penalty_per_cell": 1.20,
    "close_now_bonus": 9.20,
    "no_lock_cap_penalty": 6.80,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "g04-precision++": {
    "setup_bonus": 4.10,
    "flex_weight": 0.29,
    "big_skip_penalty_per_cell": 1.35,
    "close_now_bonus": 10.80,
    "no_lock_cap_penalty": 7.80,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "g04-anti3lock++": {
    "setup_bonus": 4.90,
    "flex_weight": 0.28,
    "big_skip_penalty_per_cell": 1.34,
    "close_now_bonus": 8.80,
    "no_lock_cap_penalty": 6.90,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "n04-anti3lockX": {
    "setup_bonus": 4.80,
    "flex_weight": 0.27,
    "big_skip_penalty_per_cell": 1.30,
    "close_now_bonus": 8.60,
    "no_lock_cap_penalty": 6.80,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  },

  "g06-anti2lockX": {
    "setup_bonus": 5.60,
    "flex_weight": 0.29,
    "big_skip_penalty_per_cell": 1.35,
    "close_now_bonus": 8.90,
    "no_lock_cap_penalty": 6.60,
    "min_component_value": 0.75,
    "combo_delta_margin": 0.25
  }
}


def ensure_profiles_manual() -> List[str]:
    """
    Register the manual HEUR profiles exactly once and persist them to disk.
    Returns the list of profile 'kinds' usable by the autoplay factory,
    e.g., ['heur_profile:n04-anti2', ...].
    """
    for name, params in MANUAL_PROFILES.items():
        ai_mod.register_heur_profile(name, params)

    # Save what we registered for reproducibility
    (OUTDIR / "profiles_manual.json").write_text(
        json.dumps({k: ai_mod.HEUR_PROFILES[k] for k in MANUAL_PROFILES.keys()}, indent=2)
    )

    return [f"heur_profile:{p}" for p in MANUAL_PROFILES.keys()]

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

    # Ensure your autoplay factory supports 'heur_profile:<name>' kinds:
    # elif kind.startswith('heur_profile:'):
    #     prof = kind.split(':', 1)[1]
    #     from .ai import HeurProfileBot
    #     return HeurProfileBot(name=name, profile=prof)

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
        "ties": report.get("ties", 0),
        "avg_scores": report["avg_scores"],
        "avg_margin": report["avg_margin"],
    }

def build_pairs(profile_kinds: List[str]) -> List[Tuple[str, str]]:
    """
    - Each manual profile vs baselines (both orders)
    - Round-robin among manual profiles (both orders)
    """
    pairs: List[Tuple[str, str]] = []
    for p in profile_kinds:
        for b in BASELINES:
            pairs.append((p, b))

    for i in range(len(profile_kinds)):
        for j in range(i + 1, len(profile_kinds)):
            pairs.append((profile_kinds[i], profile_kinds[j]))

    return pairs

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

def main():
    # 1) Register manual HEUR profiles
    prof_kinds = ensure_profiles_manual()

    # 2) Build match list
    pairs = build_pairs(prof_kinds)

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

    # 5) Print compact Win Rate vs Field (corrected A/B mapping)
    agg = accumulate_wins_and_games(results)

    # Print profiles-only
    print("\n=== HEUR profiles: Win Rate vs Field (corrected) ===")
    prof_agg = {k: v for k, v in agg.items() if k.startswith("heur_profile:")}
    rank = sorted(
        ((k, v['wins'] / v['games'] if v['games'] else 0.0, v['wins'], v['games'])
         for k, v in prof_agg.items()),
        key=lambda t: -t[1]
    )
    for k, wr, w, g in rank:
        print(f"{k:>28} WR={wr*100:5.1f}% (wins={w}, games={g})")

    # GAP baselines, just like your G script
    print("\n=== GAP baselines: Win Rate vs Profiles (this tournament) ===")
    gaps = {k: v for k, v in agg.items() if k in BASELINES}
    rank_g = sorted(
        ((k, v['wins'] / v['games'] if v['games'] else 0.0, v['wins'], v['games'])
         for k, v in gaps.items()),
        key=lambda t: -t[1]
    )
    for k, wr, w, g in rank_g:
        print(f"{k:>16} WR={wr*100:5.1f}% (wins={w}, games={g})")

    # Sanity: total wins must equal total games - total ties
    total_games = sum(r['games'] for r in results)
    total_ties = sum(r.get('ties', 0) for r in results)
    wins_sum = 0
    for r in results:
        if r['order'] == 'AB':
            wins_sum += r['wins'].get(f"A_{r['kindA']}", 0) + r['wins'].get(f"B_{r['kindB']}", 0)
        else:
            wins_sum += r['wins'].get(f"A_{r['kindB']}", 0) + r['wins'].get(f"B_{r['kindA']}", 0)
    assert wins_sum == (total_games - total_ties), \
        f"inconsistent: wins={wins_sum} vs games-ties={total_games-total_ties}"

if __name__ == "__main__":
    main()

