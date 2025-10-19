# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
import argparse
from typing import List, Dict
from .game import QwixxGame, Player, GameError, Phase, GameStatus
from .render import (
    render_player_sheet, render_score, render_roll, suggest_white_sum_targets,
    render_color_combos, COLOR_SHORT, SHORT_TO_COLOR, COLORS, BOLD, RESET
)
from .ai import HeuristicBot, MCTSBot
from .autoplay import play_series

HELP = f"""
Commands (hotkeys)
==================
  r  : roll dice (active player, initial phase)
  v  : view all boards & scores
  h  : help (works but is hidden in prompts)
  q  : quit (works but is hidden in prompts)

Bots:
  --bot  <Name>   Heuristic bot for <Name>
  --mcts <Name>   MCTS bot for <Name>

Autoplay:
  --autoplay N        Run N headless games and print a report
  --seed S            Seed for reproducible series
  --mcts-sims K       MCTS simulations per turn (default 400)
  --mcts-depth D      MCTS rollout depth in turns (default 2)
  --show-each         Print one line per game (winner & scores)
  --progress-interval Seconds between live progress updates (default 0.5)
"""

def print_header(g: QwixxGame):
    print("\n" + '='*70)
    print(f"Game {g.game_id}  |  status={g.status.value}  phase={g.phase.value}  current={g.current_player}")
    print('-'*70)

def show_all(g: QwixxGame):
    print_header(g)
    print(render_roll(g))
    print('-'*70)
    for p in g.players:
        print(f"{BOLD}{p.name}{RESET}  {render_score(g, p)}")
        print(render_player_sheet(g, p))
        print('-'*70)

# (interactive handlers unchanged from previous version)
# -- snip --  (Keep your existing handle_white_phase, handle_color_phase, prompt)

def main(argv: List[str]):
    parser = argparse.ArgumentParser(prog="qwixx.ui_cli", add_help=False)
    parser.add_argument("--bot", action="append", default=[], help="Name of a player to control with HeuristicBot (repeatable)")
    parser.add_argument("--mcts", action="append", default=[], help="Name of a player to control with MCTSBot (repeatable)")
    parser.add_argument("--autoplay", type=int, default=0, help="Run N headless games and print report")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible autoplay series")
    parser.add_argument("--mcts-sims", type=int, default=400, help="Override MCTS simulations")
    parser.add_argument("--mcts-depth", type=int, default=2, help="Override MCTS rollout depth in turns")
    parser.add_argument("--show-each", action="store_true", help="Show per-game result lines in autoplay")
    parser.add_argument("--progress-interval", type=float, default=0.5, help="Seconds between progress updates (autoplay)")
    parser.add_argument("names", nargs="*", help="Player names in turn order")
    args, _ = parser.parse_known_args(argv[1:])

    # Autoplay mode (headless report)
    if args.autoplay > 0:
        if not args.names:
            print("Error: provide player names for autoplay (e.g.,: Heur Mcts)")
            return
        names = args.names
        # Build bot types mapping
        bot_types: Dict[str, str] = {}
        for n in names:
            if n in (args.mcts or []):
                bot_types[n] = 'mcts'
            else:
                bot_types[n] = 'heur' if (args.bot and n in args.bot) or (not args.mcts) else 'heur'
        # Quick sanity: ensure every player is in either list
        missing = [n for n in names if n not in bot_types]
        if missing:
            print(f"Error: missing bot type for players: {missing}")
            return

        report = play_series(
            names=names,
            bot_types=bot_types,
            games=args.autoplay,
            mcts_sims=args.mcts_sims,
            mcts_depth=args.mcts_depth,
            seed=args.seed,
            show_each=args.show_each,
            progress_interval=args.progress_interval
        )
        print("\n=== Autoplay Report ===")
        print(f"Games: {report['games']}")
        print(f"Wins : {report['wins']}  |  Ties: {report['ties']}")
        print(f"Avg margin: {report['avg_margin']:.2f}")
        print("Avg scores:")
        for n, sc in report['avg_scores'].items():
            print(f"  - {n}: {sc:.2f}")
        return

    # --- Interactive mode below (unchanged from your last version) ---
    print("Qwixx (CLI+) - user-friendly UI")
    if args.names:
        names = args.names
    else:
        try:
            n1 = input('Player 1 name > ').strip() or 'Alice'
            n2 = input('Player 2 name > ').strip() or 'Bob'
            names = [n1, n2]
        except KeyboardInterrupt:
            print('\nBye!'); return

    players = [Player(n, is_host=(i==0)) for i, n in enumerate(names)]
    g = QwixxGame(game_id='local', players=players, current_player=players[0].name)
    g.select_first_player(players[0].name)

    bots: Dict[str, object] = {}
    for n in args.bot or []:
        if n not in names:
            print(f"Warning: --bot {n} is not in players {names}; ignoring.")
        else:
            bots[n] = HeuristicBot(n)
            print(f"Heuristic BOT enabled for player: {n}")
    for n in args.mcts or []:
        if n not in names:
            print(f"Warning: --mcts {n} is not in players {names}; ignoring.")
        else:
            bots[n] = MCTSBot(n, simulations=args.mcts_sims, depth_turns=args.mcts_depth)
            print(f"MCTS BOT enabled for player: {n}")

    # Import your previously defined prompt/handlers here or keep the definitions above.
    from .ui_cli import prompt as interactive_prompt  # if you keep handlers in this file, call prompt directly
    interactive_prompt(g, bots)

if __name__ == '__main__':
    main(sys.argv)
