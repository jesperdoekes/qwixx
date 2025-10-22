# qwixx/bootstrap_qwixx.py

from .autoplay import play_series

def run_autoplay():
    print("🚀 Running autoplay series: LearnerBot vs MCTSBot")
    report = play_series(
        names=['L', 'M'],
        bot_types={'L': 'learn', 'M': 'mcts'},
        games=50,
        epsilon=0.1,
        seed=123,
        show_each=True
    )
    print("\n=== Autoplay Report ===")
    print(f"Games: {report['games']}")
    print(f"Wins : {report['wins']}")
    print(f"Ties : {report['ties']}")
    print(f"Avg margin: {report['avg_margin']:.2f}")
    print("Avg scores:")
    for name, score in report['avg_scores'].items():
        print(f" - {name}: {score:.2f}")

if __name__ == '__main__':
    print("✅ bootstrap_qwixx.py loaded")
    run_autoplay()
