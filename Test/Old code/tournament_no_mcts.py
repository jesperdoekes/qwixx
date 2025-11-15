#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import itertools, json
from typing import Dict, List

from qwixx.autoplay import play_series

# Bots to include (no 'mcts' here)
BOT_KINDS = [
    "heur", "learn",
    "gap1", "gap1lock", "gap2", "gap2lock", "gap3", "gap3lock",
    "probgap", "probgaplock",
    "scout"
]

GAMES = 500
SEED_BASE = 4242
LEARN_FILE = "./qwixx_data/learner.json"
OUTDIR = Path("./tournament_out_no_mcts")
OUTDIR.mkdir(parents=True, exist_ok=True)

def make_bot_types(names: List[str], kinds: List[str]) -> Dict[str, str]:
    return {names[0]: kinds[0], names[1]: kinds[1]}

def run_pair(kindA: str, kindB: str, order: str, idx: int):
    if order == "AB":
        names = [f"A_{kindA}", f"B_{kindB}"]
        kinds = [kindA, kindB]
    else:
        names = [f"A_{kindB}", f"B_{kindA}"]
        kinds = [kindB, kindA]

    bot_types = make_bot_types(names, kinds)
    seed = SEED_BASE + idx
    report = play_series(
        names=names,
        bot_types=bot_types,
        games=GAMES,
        mcts_sims=200,      # ignored (no mcts present)
        mcts_depth=2,
        seed=seed,
        show_each=False,
        progress_interval=0.5,
        epsilon=0.10,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        learn_file=LEARN_FILE,
        debug_timing=False
    )
    row = {
        "pair_index": idx,
        "kindA": kindA, "kindB": kindB, "order": order,
        "games": report["games"],
        "wins": report["wins"],
        "ties": report["ties"],
        "avg_scores": report["avg_scores"],
        "avg_margin": report["avg_margin"],
    }
    return row

def main():
    results = []
    idx = 0
    for i, j in itertools.combinations_with_replacement(range(len(BOT_KINDS)), 2):
        A = BOT_KINDS[i]; B = BOT_KINDS[j]
        for order in ("AB", "BA"):
            idx += 1
            print(f"=== Pair {idx}: {A} vs {B} [order={order}] ===")
            rep = run_pair(A, B, order, idx)
            results.append(rep)
            print(json.dumps({
                "A": A, "B": B, "order": order,
                "wins": rep["wins"], "ties": rep["ties"],
                "avg_margin": rep["avg_margin"]
            }, indent=2))
            print()

    out_json = OUTDIR / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
