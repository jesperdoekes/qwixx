# run_gap_bots.py
from qwixx.autoplay import play_series

def do_pair(n1, f1, n2, f2, games=50, seed=42, mcts_sims=400, mcts_depth=2):
    bot_types = {n1: f1, n2: f2}
    report = play_series(
        names=[n1, n2],
        bot_types=bot_types,
        games=games,
        mcts_sims=mcts_sims,
        mcts_depth=mcts_depth,
        seed=seed,
        show_each=False,          # set True if you want a line per game
        progress_interval=0.1,    # fast updates
        epsilon=0.10,             # only used if a Learner is in the match
        epsilon_end=0.05,
        epsilon_decay=0.995,
        learn_file="./qwixx_data/learner.json",
        debug_timing=False
    )
    return report

def print_report(title, r):
    print(f"\n=== {title} ===")
    print(f"Games: {r['games']}")
    print(f"Wins : {r['wins']} | Ties: {r['ties']}")
    print(f"Avg margin: {r['avg_margin']:.2f}")
    print("Avg scores:")
    for name, sc in sorted(r['avg_scores'].items()):
        print(f" - {name}: {sc:.2f}")

def main():
    # Internal labels used in summary
    # Mapping to autoplay bot_types keys (must match your qwixx.autoplay implementation):
    #   'gap1' | 'gap1lock' | 'gap2' | 'gap2lock'
    pairs = [
        ("G1",  "gap1",     "G1L", "gap1lock"),
        ("G1",  "gap1",     "G2",  "gap2"),
        ("G1",  "gap1",     "G2L", "gap2lock"),
        ("G1L", "gap1lock", "G2",  "gap2"),
        ("G1L", "gap1lock", "G2L", "gap2lock"),
        ("G2",  "gap2",     "G2L", "gap2lock"),
    ]

    all_reports = []
    for (n1, f1, n2, f2) in pairs:
        title = f"{n1} ({f1}) vs {n2} ({f2})"
        report = do_pair(n1, f1, n2, f2, games=50, seed=42)
        print_report(title, report)
        all_reports.append((title, report))

    # Optional: overall summary aggregation across all six pairings
    print("\n====== Overall Summary (6 pairings, 50 games each) ======")
    total_games = sum(r['games'] for _, r in all_reports)
    total_wins = {}
    total_ties = sum(r['ties'] for _, r in all_reports)
    for _, r in all_reports:
        for k, v in r['wins'].items():
            total_wins[k] = total_wins.get(k, 0) + v
    print(f"Total games: {total_games} | Total ties: {total_ties}")
    print("Total wins across all pairings:")
    for k in sorted(total_wins.keys()):
        print(f" - {k}: {total_wins[k]}")

if __name__ == "__main__":
    main()
