#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json
from typing import Dict, List, Tuple
from qwixx.autoplay import play_series

G_FILE = Path("./tournament_heur_G_out/heur_G_results.json")
OUTDIR = Path("./tournament_gap_all_short_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Keep it short but meaningful
GAMES = 100
ORDERS = ("AB",)   # single order to cut runtime; switch to ("AB","BA") if you want symmetry
SEED_BASE = 20251

# Classic + Adaptive + Probability gap family
GAP_KINDS = [
    "gap1", "gap1lock",
    "gap2", "gap2lock",
    "gap3", "gap3lock",
    "gap_adapt1", "gap_adapt1lock",
    "gap_adapt2", "gap_adapt2lock",
    "gap_adapt3", "gap_adapt3lock",
    "probgap", "probgaplock",
    "probgap2", "probgaplock2",
]

def corrected_wr_agg(results: List[Dict]) -> Dict[str, Dict[str,int]]:
    agg: Dict[str, Dict[str,int]] = {}
    def bump(bot, w, g):
        a = agg.setdefault(bot, {'wins':0,'games':0}); a['wins'] += w; a['games'] += g
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

def top2_profiles_from_G(g_file: Path) -> List[str]:
    data = json.loads(g_file.read_text())
    agg = corrected_wr_agg(data)
    prof = {k:v for k,v in agg.items() if k.startswith("heur_profile:")}
    if not prof:
        raise RuntimeError("No heur_profile:* in tournament_heur_G_out/heur_G_results.json")
    ranked = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0) for k,v in prof.items()),
                    key=lambda t: -t[1])
    return [ranked[0][0], ranked[1][0]] if len(ranked)>=2 else [ranked[0][0]]

def run_pair(kindA: str, kindB: str, order: str, idx: int, games: int) -> Dict:
    if order == "AB":
        names = [f"A_{kindA}", f"B_{kindB}"]; kinds = [kindA, kindB]
    else:
        names = [f"A_{kindB}", f"B_{kindA}"]; kinds = [kindB, kindA]
    rep = play_series(
        names=names, bot_types={names[0]:kinds[0], names[1]:kinds[1]},
        games=games, mcts_sims=0, mcts_depth=0,
        seed=SEED_BASE+idx, show_each=False, progress_interval=0.5,
        epsilon=0.10, epsilon_end=0.05, epsilon_decay=0.995,
        learn_file="./qwixx_data/learner.json", debug_timing=False
    )
    return {
        "pair_index": idx, "kindA": kindA, "kindB": kindB, "order": order,
        "games": rep["games"], "wins": rep["wins"], "ties": rep["ties"],
        "avg_scores": rep["avg_scores"], "avg_margin": rep["avg_margin"],
    }

def main():
    # Pool: all gap bots + heur_strong + Top-2 from G
    top2 = top2_profiles_from_G(G_FILE)
    pool = GAP_KINDS + ["heur_strong"] + top2
    print("Pool size:", len(pool))
    print("Top-2 from G:", top2)

    # Short round-robin among the pool
    results = []
    idx = 0
    for i in range(len(pool)):
        for j in range(i+1, len(pool)):
            A, B = pool[i], pool[j]
            for order in ORDERS:
                idx += 1
                print(f"=== Pair {idx}: {A} vs {B} [{order}] ===")
                results.append(run_pair(A,B,order,idx,GAMES))

    # Save
    out_json = OUTDIR / "gap_all_short_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print("Saved:", out_json)

    # Print quick Win% table
    agg = corrected_wr_agg(results)
    ranked = sorted(((k, v['wins']/v['games'] if v['games'] else 0.0, v['wins'], v['games']) for k,v in agg.items()),
                    key=lambda t: -t[1])
    print("\n=== Short tournament: Win Rate vs Field ===")
    for k, wr, w, g in ranked:
        print(f"{k:>24}  WR={wr*100:5.1f}%  (wins={w}, games={g})")

if __name__ == "__main__":
    main()
