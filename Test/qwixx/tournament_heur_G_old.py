#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json
from typing import Dict, List, Tuple
from qwixx.autoplay import play_series
from qwixx import ai as ai_mod

# Inputs from previous run
#PREV_OUT = Path("./tournament_heur_test_out/heur_test_results.json")
#PREV_PROFILES = Path("./tournament_heur_test_out/profiles.json")

PREV_OUT = Path("./tournament_heur_G_out (4)/heur_G_results.json")
PREV_PROFILES = Path("./tournament_heur_G_out (4)/profiles.json")

# Outputs for this run
OUTDIR = Path("./tournament_heur_G_out (5)")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED_BASE = 24684
GAMES = 400
BASELINES = ["gap1", "gap1lock", "gap2", "gap2lock"]

# Ranges for the 6 new randoms
RANGES = {
    'setup_bonus': (3.8, 6.0),
    'flex_weight': (0.14, 0.26),
    'big_skip_penalty_per_cell': (0.95, 1.30),
    'close_now_bonus': (8.0, 11.0),
    'no_lock_cap_penalty': (5.5, 8.0),
    'min_component_value': (ai_mod.MIN_COMPONENT_VALUE, ai_mod.MIN_COMPONENT_VALUE),
    'combo_delta_margin': (ai_mod.COMBO_DELTA_MARGIN, ai_mod.COMBO_DELTA_MARGIN),
}

def corrected_wr_agg(results: List[Dict]) -> Dict[str, Dict[str,int]]:
    """Correctly accumulate wins/games with AB/BA mapping."""
    agg: Dict[str, Dict[str,int]] = {}
    def bump(bot, w, g):
        a = agg.setdefault(bot, {'wins':0,'games':0})
        a['wins'] += w; a['games'] += g

    for r in results:
        g = r['games']
        if r['order'] == 'AB':
            a_bot, b_bot = r['kindA'], r['kindB']
            w_a = r['wins'].get(f"A_{a_bot}", 0)
            w_b = r['wins'].get(f"B_{b_bot}", 0)
        else:
            a_bot, b_bot = r['kindB'], r['kindA']
            w_a = r['wins'].get(f"A_{a_bot}", 0)
            w_b = r['wins'].get(f"B_{b_bot}", 0)
        bump(a_bot, w_a, g)
        bump(b_bot, w_b, g)
    return agg

def pick_top2_profiles(prev_results_file: Path) -> List[str]:
    """Return ['heur_profile:<name>', ...] of top-2 HEUR profiles by Win% from previous run."""
    if not prev_results_file.exists():
        raise FileNotFoundError(f"Missing previous results: {prev_results_file}")
    results = json.loads(prev_results_file.read_text())
    agg = corrected_wr_agg(results)
    prof = {k:v for k,v in agg.items() if k.startswith("heur_profile:")}
    if not prof:
        raise RuntimeError("No heur_profile:* entries found in previous results.")
    ranked = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0) for k,v in prof.items()),
                    key=lambda t: -t[1])
    top2 = [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else [ranked[0][0]]
    return top2

def load_prev_profile_params() -> Dict[str, Dict[str, float]]:
    """Load name->params (e.g. {'rand07': {...}}) from previous tournament profiles.json."""
    if not PREV_PROFILES.exists():
        return {}
    try:
        return json.loads(PREV_PROFILES.read_text())
    except Exception:
        return {}

def ensure_registered(names: List[str]):
    """
    Ensure each heur_profile:<name> is present in ai_mod.HEUR_PROFILES.
    If missing, load from PREV_PROFILES and register.
    """
    prev_params = load_prev_profile_params()
    missing: List[str] = []
    for full in names:
        if not full.startswith("heur_profile:"):
            continue
        key = full.split(":", 1)[1]  # e.g., 'rand07'
        if key not in ai_mod.HEUR_PROFILES:
            params = prev_params.get(key)
            if params is None:
                missing.append(key)
            else:
                ai_mod.register_heur_profile(key, params)
    if missing:
        raise KeyError(
            f"Missing profiles in registry and not found in {PREV_PROFILES}: {missing}. "
            f"Make sure previous tournament saved profiles.json with these names."
        )

def register_six_random(prefix="g", seed=SEED_BASE) -> List[str]:
    names = ai_mod.bulk_register_random_profiles(
        count=6, seed=seed, ranges=RANGES, prefix=prefix
    )
    # return fully-qualified kinds
    return [f"heur_profile:{n}" for n in names]

def run_pair(kindA: str, kindB: str, order: str, idx: int, games: int) -> Dict:
    if order == "AB":
        names = [f"A_{kindA}", f"B_{kindB}"]; kinds = [kindA, kindB]
    else:
        names = [f"A_{kindB}", f"B_{kindA}"]; kinds = [kindB, kindA]
    report = play_series(
        names=names, bot_types={names[0]:kinds[0], names[1]:kinds[1]},
        games=games, mcts_sims=0, mcts_depth=0,
        seed=SEED_BASE+idx, show_each=False, progress_interval=0.5,
        epsilon=0.10, epsilon_end=0.05, epsilon_decay=0.995,
        learn_file="./qwixx_data/learner.json", debug_timing=False
    )
    return {
        "pair_index": idx, "kindA": kindA, "kindB": kindB, "order": order,
        "games": report["games"], "wins": report["wins"], "ties": report["ties"],
        "avg_scores": report["avg_scores"], "avg_margin": report["avg_margin"],
    }

def main():
    # 1) Top-2 from previous tournament (by Win% using corrected mapping)
    top2 = pick_top2_profiles(PREV_OUT)
    print("Top-2 from previous:", top2)

    # 2) Ensure their params are registered (load from previous profiles.json if needed)
    ensure_registered(top2)

    # 3) Register six new random profiles
    new6 = register_six_random(prefix="g")
    print("New random profiles:", new6)

    # 4) Build pool and save their params (Top-2 + new6)
    pool = top2 + new6
    # Extract params from registry
    profiles_dict = {}
    for k in pool:
        if not k.startswith("heur_profile:"):
            continue
        key = k.split(":", 1)[1]
        profiles_dict[key] = ai_mod.HEUR_PROFILES[key]
    (OUTDIR / "profiles.json").write_text(json.dumps(profiles_dict, indent=2))

    # 5) Build match list (vs  + round-robin among the 8)
    pairs: List[Tuple[str,str]] = []
    for p in pool:
        for b in BASELINES:
            pairs.append((p, b))
    for i in range(len(pool)):
        for j in range(i+1, len(pool)):
            pairs.append((pool[i], pool[j]))

    # 6) Run
    results = []
    idx = 0
    for A,B in pairs:
        for order in ("AB","BA"):
            idx += 1
            print(f"=== Pair {idx}: {A} vs {B} [{order}] ===")
            results.append(run_pair(A,B,order,idx,GAMES))

    # 7) Save & print quick table for profiles only
    out_json = OUTDIR / "heur_G_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print("Saved:", out_json)

    agg = corrected_wr_agg(results)
    prof = {k:v for k,v in agg.items() if k.startswith("heur_profile:")}
    ranked = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0, v['wins'], v['games']) for k,v in prof.items()),
                    key=lambda t: -t[1])
    print("\n=== HEUR G tournament: Win Rate vs Field ===")
    for k, wr, w, g in ranked:
        print(f"{k:>28}  WR={wr*100:5.1f}%  (wins={w}, games={g})")

if __name__ == "__main__":
    main()
