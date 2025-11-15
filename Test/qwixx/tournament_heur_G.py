#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json, random
from typing import Dict, List, Tuple, Optional
from qwixx.autoplay import play_series
from qwixx import ai as ai_mod

# === Paths ===
PREV_DIR = Path("./tournament_heur_G_out (6)")
PREV_RESULTS = PREV_DIR / "heur_G_results.json"
PREV_PROFILES = PREV_DIR / "profiles.json"

OUTDIR = Path("./tournament_heur_G_out (7)")
OUTDIR.mkdir(parents=True, exist_ok=True)

# === Config ===
SEED_BASE = 24684
GAMES = 400
# Expand baselines to include the extra GAPs you requested
BASELINES = [
#    "gap1", "gap1lock",
    "gap2", "gap2lock",
    # added:
    "gap3", "gap3lock",
    "gap_adapt3", "gap_adapt3lock",
]

# Ranges for new randoms
RANGES = {
    'setup_bonus': (3.8, 6.0),
    'flex_weight': (0.14, 0.26),
    'big_skip_penalty_per_cell': (0.95, 1.30),
    'close_now_bonus': (8.0, 11.0),
    'no_lock_cap_penalty': (5.5, 8.0),
    'min_component_value': (ai_mod.MIN_COMPONENT_VALUE, ai_mod.MIN_COMPONENT_VALUE),
    'combo_delta_margin': (ai_mod.COMBO_DELTA_MARGIN, ai_mod.COMBO_DELTA_MARGIN),
}

NEW_COUNT = 6
NEW_PREFIX = "new_"   # <-- prevents collision with old 'g##'
OLD_PREFIX = "old_"   # <-- winners cloned under 'old_...'

# ---------- Utilities ----------
def corrected_wr_agg(results: List[Dict]) -> Dict[str, Dict[str,int]]:
    """Aggregate wins/games for each kind, respecting AB vs BA rows."""
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
    """Return top-2 kind strings like 'heur_profile:g05' from previous G results."""
    if not prev_results_file.exists():
        raise FileNotFoundError(f"Missing previous results: {prev_results_file}")
    results = json.loads(prev_results_file.read_text())
    agg = corrected_wr_agg(results)
    prof = {k:v for k,v in agg.items() if k.startswith("heur_profile:")}
    if not prof:
        raise RuntimeError("No heur_profile:* entries found in previous results.")
    ranked = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0) for k,v in prof.items()),
                    key=lambda t: -t[1])
    return [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else [ranked[0][0]]

def load_prev_profiles(prev_profiles_file: Path) -> Dict[str, Dict[str,float]]:
    """
    Load params from previous profiles.json.
    We accept keys either as 'heur_profile:<raw>' or just '<raw>'.
    Returns map raw_name -> params.
    """
    if not prev_profiles_file.exists():
        print(f"[Warn] {prev_profiles_file} not found; will attempt to clone from in-memory HEUR_PROFILES if available.")
        return {}
    data = json.loads(prev_profiles_file.read_text())
    out: Dict[str, Dict[str,float]] = {}
    for k, params in data.items():
        raw = k.split(':',1)[1] if k.startswith("heur_profile:") else k
        out[raw] = params
    return out

def register_old_clone(kind: str, prev_map: Dict[str, Dict[str,float]]) -> str:
    """
    Given kind string 'heur_profile:<name>', clone its params to 'old_<name>' in registry.
    Return new kind string 'heur_profile:old_<name>'.
    """
    assert kind.startswith("heur_profile:")
    raw = kind.split(':',1)[1]
    params = prev_map.get(raw) or ai_mod.HEUR_PROFILES.get(raw)
    if not params:
        raise RuntimeError(f"Params for {raw} not found in previous profiles nor current registry.")
    new_raw = f"{OLD_PREFIX}{raw}"
    ai_mod.register_heur_profile(new_raw, params)
    return f"heur_profile:{new_raw}"

def register_new_randoms(count: int = NEW_COUNT, seed: int = SEED_BASE) -> List[str]:
    names = ai_mod.bulk_register_random_profiles(
        count=count, seed=seed, ranges=RANGES, prefix=NEW_PREFIX
    )
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

# ---------- Main ----------
def main():
    # 1) Pick Top-2 from prior results & load their params
    top2_prev = pick_top2_profiles(PREV_RESULTS)  # e.g. ['heur_profile:g05','heur_profile:g06']
    prev_param_map = load_prev_profiles(PREV_PROFILES)

    # 2) Clone winners under 'old_<name>' to avoid collisions
    top2_old = [register_old_clone(k, prev_param_map) for k in top2_prev]
    print("Top-2 cloned as:", top2_old)

    # 3) Register six new profiles under 'new_' prefix (no collision with prior 'g##')
    new6 = register_new_randoms()
    print("New 6 profiles:", new6)

    # 4) Save current registry params for the pool (old_ + new_)
    pool = top2_old + new6
    (OUTDIR / "profiles.json").write_text(json.dumps(
        {p: ai_mod.HEUR_PROFILES[p.split(':',1)[1]] for p in pool}, indent=2
    ))

    # 5) Build match list:
    #    - each pool profile vs (expanded) GAP baselines (AB+BA)
    #    - round-robin among the pool profiles (AB+BA)
    pairs: List[Tuple[str,str]] = []
    for p in pool:
        for b in BASELINES:
            pairs.append((p, b))
    for i in range(len(pool)):
        for j in range(i+1, len(pool)):
            pairs.append((pool[i], pool[j]))

    # 6) Run
    results: List[Dict] = []
    idx = 0
    for A,B in pairs:
        for order in ("AB","BA"):
            idx += 1
            print(f"=== Pair {idx}: {A} vs {B} [{order}] ===")
            results.append(run_pair(A,B,order,idx,GAMES))

    # 7) Save results
    out_json = OUTDIR / "heur_G_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print("Saved:", out_json)

    # 8) Reporting — Profiles AND GAP baselines
    agg = corrected_wr_agg(results)

    # a) Profiles Win% (vs field)
    print("\n=== HEUR profiles: Win Rate vs Field ===")
    prof = {k:v for k,v in agg.items() if k.startswith("heur_profile:")}
    ranked = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0, v['wins'], v['games']) for k,v in prof.items()),
                    key=lambda t: -t[1])
    for k, wr, w, g in ranked:
        print(f"{k:>28}  WR={wr*100:5.1f}%  (wins={w}, games={g})")

    # b) GAP baselines Win% (aggregate vs the pool of profiles)
    print("\n=== GAP baselines: Win Rate vs Profiles (this tournament) ===")
    gaps = {k:v for k,v in agg.items() if k in BASELINES}
    ranked_g = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0, v['wins'], v['games']) for k,v in gaps.items()),
                      key=lambda t: -t[1])
    for k, wr, w, g in ranked_g:
        print(f"{k:>16}  WR={wr*100:5.1f}%  (wins={w}, games={g})")

    # c) Focus — print the four you asked to add
    focus = ["gap_adapt3lock", "gap_adapt3", "gap3", "gap3lock"]
    print("\n=== Focus GAPs (requested) ===")
    for k in focus:
        s = gaps.get(k, {'wins':0,'games':0})
        wr = (s['wins']/s['games']) if s['games'] else 0.0
        print(f"{k:>16}  WR={wr*100:5.1f}%  (wins={s['wins']}, games={s['games']})")

if __name__ == "__main__":
    main()
