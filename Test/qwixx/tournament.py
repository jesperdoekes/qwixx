#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tournament runner with resume/slice support.

New CLI:
  --start-pair N   : start from this 1-based pair index (matches log numbering)
  --end-pair M     : stop at this 1-based pair index (inclusive)
  --resume         : skip pairs already present in tournament_results.json, continue with the rest

Pair ordering is identical to prior runs:
  for each unordered pair (with replacement) from BOT_KINDS:
    run order="AB" then order="BA"
This yields 56 pairs when BOT_KINDS has 7 kinds.

Outputs:
  tournament_out/tournament_results.json  (appended/merged)
  tournament_out/tournament_results.csv   (rewritten on each run)
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

from qwixx.autoplay import play_series

# ------------------ config ------------------
BOT_KINDS = ["heur", "mcts", "learn", "gap1", "gap1lock", "gap2", "gap2lock"]

GAMES = 50
SEED_BASE = 42
LEARN_FILE = "./qwixx_data/learner.json"
OUTDIR = Path("./tournament_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUTDIR / "tournament_results.json"
CSV_PATH = OUTDIR / "tournament_results.csv"

# -------------- helpers ---------------------
def enumerate_pairs() -> List[Tuple[str, str, str]]:
    """
    Produce the full list of (kindA, kindB, order) in exactly the same order
    you used before:

      for (A,B) in combinations_with_replacement(BOT_KINDS, 2):
          order in ("AB","BA")

    The returned list is indexed 0-based internally; we display 1-based pair numbers.
    """
    lst: List[Tuple[str, str, str]] = []
    for i, j in itertools.combinations_with_replacement(range(len(BOT_KINDS)), 2):
        A = BOT_KINDS[i]
        B = BOT_KINDS[j]
        lst.append((A, B, "AB"))
        lst.append((A, B, "BA"))
    return lst

def load_existing() -> List[Dict]:
    if JSON_PATH.exists():
        with JSON_PATH.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    return []

def save_json(results: List[Dict]):
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=False)

def write_csv(results: List[Dict]):
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("pair,kindA,kindB,order,games,ties,avg_margin,"
                "wins_A_name,wins_A_count,wins_B_name,wins_B_count,avgA,avgB\n")
        for item in results:
            pair = item.get("pair_index", "?")
            kindA = item["kindA"]; kindB = item["kindB"]; order = item["order"]
            games = item["games"]; ties = item["ties"]; avg_margin = item["avg_margin"]
            wins = item["wins"]; avg_scores = item["avg_scores"]

            # Reconstruct names based on order (first name is starter 'A_*')
            if order == "AB":
                nameA = f"A_{kindA}"
                nameB = f"B_{kindB}"
            else:
                nameA = f"A_{kindB}"
                nameB = f"B_{kindA}"

            wA = wins.get(nameA, 0)
            wB = wins.get(nameB, 0)
            avgA = avg_scores.get(nameA, 0.0)
            avgB = avg_scores.get(nameB, 0.0)

            f.write(f"{pair},{kindA},{kindB},{order},"
                    f"{games},{ties},{avg_margin:.3f},"
                    f"{nameA},{wA},{nameB},{wB},{avgA:.2f},{avgB:.2f}\n")

def result_key(item: Dict) -> Tuple[str, str, str]:
    """Uniqueness key for a pairing row."""
    return (item["kindA"], item["kindB"], item["order"])

def make_bot_types(names: List[str], kinds: List[str]) -> Dict[str, str]:
    assert len(names) == len(kinds) == 2
    return {names[0]: kinds[0], names[1]: kinds[1]}

def run_one_pair(pair_idx_1based: int, kindA: str, kindB: str, order: str) -> Dict:
    """Run a single 50-game pairing for (kindA, kindB, order)."""
    # Names reflect order: first name starts (A_*), second follows (B_*)
    if order == "AB":
        names = [f"A_{kindA}", f"B_{kindB}"]
        kinds = [kindA, kindB]
    else:
        names = [f"A_{kindB}", f"B_{kindA}"]
        kinds = [kindB, kindA]

    bot_types = make_bot_types(names, kinds)
    seed = SEED_BASE + pair_idx_1based  # reproducible but distinct per pair

    report = play_series(
        names=names,
        bot_types=bot_types,
        games=GAMES,
        mcts_sims=400,
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
        "pair_index": pair_idx_1based,
        "kindA": kindA,
        "kindB": kindB,
        "order": order,
        "games": report["games"],
        "wins": report["wins"],
        "ties": report["ties"],
        "avg_scores": report["avg_scores"],
        "avg_margin": report["avg_margin"],
    }
    return row

def main():
    ap = argparse.ArgumentParser(description="Qwixx tournament runner with resume/slice")
    ap.add_argument("--start-pair", type=int, default=None, help="1-based pair index to start at")
    ap.add_argument("--end-pair", type=int, default=None, help="1-based pair index to stop at (inclusive)")
    ap.add_argument("--resume", action="store_true", help="Skip pairs already present in results JSON and continue with remaining")
    args = ap.parse_args()

    pairs = enumerate_pairs()  # 56 items for 7 bots (28 unordered × 2 orders)
    total_pairs = len(pairs)

    # Determine slice by indices (1-based external, 0-based internal)
    start_1 = args.start_pair if args.start_pair is not None else 1
    end_1 = args.end_pair if args.end_pair is not None else total_pairs
    if start_1 < 1 or end_1 < start_1 or end_1 > total_pairs:
        raise SystemExit(f"Invalid slice: start={start_1}, end={end_1}, total={total_pairs}")

    # Load existing results (if any)
    existing = load_existing()
    existing_by_key = {result_key(x): x for x in existing}

    # If resuming, skip any pair already in existing
    run_results: List[Dict] = existing[:]  # start from existing to preserve prior rows

    for idx0 in range(start_1 - 1, end_1):
        pair_idx_1based = idx0 + 1
        kindA, kindB, order = pairs[idx0]
        key = (kindA, kindB, order)

        if args.resume and key in existing_by_key:
            # Already completed — skip
            print(f"=== Pair {pair_idx_1based}: {kindA} vs {kindB} [order={order}] ===")
            print("SKIP (already in results)\n")
            continue

        print(f"=== Pair {pair_idx_1based}: {kindA} vs {kindB} [order={order}] ===")
        row = run_one_pair(pair_idx_1based, kindA, kindB, order)
        run_results.append(row)

        # Persist after each pairing to protect progress
        save_json(run_results)
        write_csv(run_results)

        # Brief console summary (mimic your earlier print)
        summary = {
            "A": kindA,
            "B": kindB,
            "order": order,
            "wins": row["wins"],
            "ties": row["ties"],
            "avg_margin": row["avg_margin"]
        }
        print(json.dumps(summary, indent=2))
        print()

    print("Done.")
    print(f"JSON: {JSON_PATH}")
    print(f"CSV : {CSV_PATH}")

if __name__ == "__main__":
    main()
