# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
import argparse
from typing import List, Dict

from .game import QwixxGame, Player, GameError, Phase, GameStatus
from .render import (
    render_player_sheet, render_score, render_roll, suggest_white_sum_targets,
    render_color_combos, COLOR_SHORT, SHORT_TO_COLOR, BOLD, RESET
)

from .ai import HeuristicBot, MCTSBot, LearnerBot, \
    Gap1Bot, Gap1LockBot, Gap2Bot, Gap2LockBot, Gap3Bot, Gap3LockBot, \
    ProbGapBot, ProbGapLockBot, ScoutBot

from .autoplay import play_series

HELP = f"""
Commands (hotkeys)
==================
 r : roll dice (active player, initial phase)
 v : view all boards & scores
 h : help
 q : quit
Bots:
 --bot <Name>         Heuristic bot
 --mcts <Name>        MCTS bot
 --learn <Name>       Learner bot (ε-greedy)
 --gap1 <Name>        Max-skip 1 bot
 --gap1-lock <Name>   Max-skip 1, but may break only to lock
 --gap2 <Name>        Max-skip 2 bot
 --gap2-lock <Name>   Max-skip 2, but may break only to lock
Learner options:
 --learn-file PATH    Path to learner JSON store (default ./qwixx_data/learner.json)
 --epsilon P          Exploration rate for learners (start, default 0.10)
Autoplay:
 --autoplay N         Run N headless games and print a report
 --seed S             Seed for reproducible series
 --mcts-sims K        MCTS simulations per turn (default 400)
 --mcts-depth D       MCTS rollout depth in turns (default 2)
 --show-each          Print one line per game (winner & scores)
 --progress-interval  Seconds between progress updates (default 0.5)
Debug:
 --debug-bots         Print players and the bot type assigned to each (autoplay)
 --debug-timing       Print per-game elapsed time (autoplay)
"""

def print_header(g: QwixxGame):
    print("\n" + '='*70)
    print(f"Game {g.game_id} \n status={g.status.value} phase={g.phase.value} current={g.current_player}")
    print('-'*70)

def show_all(g: QwixxGame):
    print_header(g)
    print(render_roll(g))
    print('-'*70)
    for p in g.players:
        print(f"{BOLD}{p.name}{RESET} {render_score(g, p)}")
        print(render_player_sheet(g, p))
        print('-'*70)

def handle_white_phase(g: QwixxGame, bots: Dict[str, object] | None = None):
    bots = bots or {}
    for p in g.players:
        if g.white_phase_complete.get(p.name, False):
            continue
        # Active bot with a plan
        if p.name in bots and p.name == g.current_player and getattr(bots[p.name], 'pending_plan', None):
            action, color = bots[p.name].pending_plan.get('white_action', ('skip', None))
            if action == 'mark' and color:
                ws = g.white_sum()
                try:
                    g.mark_number_white_sum(p.name, color, ws)
                    print(f"{p.name} [BOT] (planned) marked white-sum {ws} on {COLOR_SHORT[color]}.")
                except GameError as e:
                    print(f"{p.name} [BOT] (planned) error: {e}")
                    g.skip_white_sum(p.name)
            else:
                try:
                    g.skip_white_sum(p.name)
                    print(f"{p.name} [BOT] (planned) skipped white-sum.")
                except GameError as e:
                    print(f"{p.name} [BOT] (planned) skip error: {e}")
            continue

        # Non-active bot
        if p.name in bots:
            action, color = bots[p.name].choose_white_sum_action(g)
            if action == 'mark' and color:
                ws = g.white_sum()
                try:
                    g.mark_number_white_sum(p.name, color, ws)
                    print(f"{p.name} [BOT] marked white-sum {ws} on {COLOR_SHORT[color]}.")
                except GameError as e:
                    print(f"{p.name} [BOT] error: {e}")
                    g.skip_white_sum(p.name)
            else:
                g.skip_white_sum(p.name)
                print(f"{p.name} [BOT] skipped white-sum.")
            continue

        # Human
        opts = suggest_white_sum_targets(g, p)
        opt_str = ("/".join(opts) or "none")
        while True:
            inp = input(f"{p.name}: mark white sum on [{opt_str}] or 'skip' > ").strip().upper()
            if inp == 'Q':
                raise KeyboardInterrupt
            if inp == 'H':
                print(HELP); continue
            if inp == '' or inp == 'SKIP':
                try:
                    g.skip_white_sum(p.name)
                except GameError as e:
                    print(f"Error: {e}")
                    continue
                break
            if inp in ('R','Y','G','B'):
                color = SHORT_TO_COLOR[inp]
                ws = g.white_sum()
                try:
                    g.mark_number_white_sum(p.name, color, ws)
                except GameError as e:
                    print(f"Error: {e}")
                    continue
                break
            print("Enter R/Y/G/B or 'skip' (or press Enter to skip).")
    g.proceed_to_color_phase_if_ready()

def handle_color_phase(g: QwixxGame, bots: Dict[str, object] | None = None):
    bots = bots or {}
    cp = g.current_player
    marked_white = bool(g.turn_actions.get(cp, {}).get('whiteSumMarked', False))

    # Active bot with planned action
    if cp in bots and getattr(bots[cp], 'pending_plan', None):
        act, color, wd = bots[cp].pending_plan.get('color_action', ('skip', None, None))
        try:
            if act == 'mark' and color and wd:
                g.select_dice_for_color_phase(cp, wd, color)
                cv = g.combined_value()
                g.mark_number_color_dice(cp, color, cv)
                print(f"{cp} [BOT] (planned) marked {cv} on {COLOR_SHORT[color]} using white{wd}.")
            else:
                if marked_white:
                    print(f"{cp} [BOT] (planned) skipped color (white-sum was marked) → no penalty.")
                else:
                    print(f"{cp} [BOT] (planned) skipped color after skipping white-sum → penalty taken.")
                    g.take_penalty(cp)
        except GameError as e:
            if not marked_white:
                print(f"{cp} [BOT] (planned) error → penalty taken: {e}")
                try: g.take_penalty(cp)
                except GameError as e2: print(f"Error: {e2}")
            else:
                print(f"{cp} [BOT] (planned) error → skip without penalty: {e}")
        finally:
            bots[cp].pending_plan = None
        return

    # Non-planned bot
    if cp in bots:
        act, color, wd = bots[cp].choose_color_action(g)
        if act == 'mark' and color and wd:
            try:
                g.select_dice_for_color_phase(cp, wd, color)
                cv = g.combined_value()
                g.mark_number_color_dice(cp, color, cv)
                print(f"{cp} [BOT] marked {cv} on {COLOR_SHORT[color]} using white{wd}.")
            except GameError as e:
                if not marked_white:
                    print(f"{cp} [BOT] error → penalty taken: {e}")
                    try: g.take_penalty(cp)
                    except GameError as e2: print(f"Error: {e2}")
                else:
                    print(f"{cp} [BOT] error → skip without penalty: {e}")
        else:
            if marked_white:
                print(f"{cp} [BOT] skipped color (white-sum was marked) → no penalty.")
            else:
                print(f"{cp} [BOT] skipped color after skipping white-sum → penalty taken.")
                try: g.take_penalty(cp)
                except GameError as e: print(f"Error: {e}")
        return

    # Human
    from .ai import legal_color_actions
    acts = legal_color_actions(g, cp)
    if not acts:
        if marked_white:
            print(f"{cp}: no legal color-dice marks → skipping color without penalty (white-sum was marked).")
            return
        else:
            print(f"{cp}: no legal color-dice marks and white-sum was skipped → automatic penalty.")
            try: g.take_penalty(cp)
            except GameError as e: print(f"Error: {e}")
            return

    print(render_color_combos(g, next(p for p in g.players if p.name == cp)))
    while True:
        sel = input(f"{cp}: choose color (R/Y/G/B) or press Enter to skip > ").strip().upper()
        if sel == 'H':
            print(HELP); continue
        if sel == 'Q':
            raise KeyboardInterrupt
        if sel == '':
            if marked_white:
                print(f"{cp} skipped color action (white-sum was marked) → no penalty.")
                return
            else:
                print(f"{cp} skipped color action after skipping white-sum → penalty taken.")
                try: g.take_penalty(cp)
                except GameError as e: print(f"Error: {e}")
                return
        if sel not in ('R','Y','G','B'):
            print("Please enter R/Y/G/B or press Enter to skip.")
            continue
        w = input("Which white die? 1 or 2 > ").strip()
        if w not in ('1','2'):
            print("Enter 1 or 2.")
            continue
        color = SHORT_TO_COLOR[sel]
        try:
            g.select_dice_for_color_phase(cp, int(w), color)
            cv = g.combined_value()
            g.mark_number_color_dice(cp, color, cv)
            print(f"{cp} marked {cv} on {sel} using white{w}.")
            return
        except GameError as e:
            print(f"Error: {e}")
            continue

def prompt(g: QwixxGame, bots: Dict[str, object]):
    show_all(g)
    while True:
        try:
            if g.status == GameStatus.FINISHED:
                print("Game finished. Thanks for playing!")
                # If a learner participated, let it record final reward
                scores = {p.name: g.calculate_score(p.name)['total'] for p in g.players}
                for b in bots.values():
                    if isinstance(b, LearnerBot):
                        b.notify_game_end(scores, save=True)
                return

            if g.phase == Phase.INITIAL:
                cmd = input("(r)oll / (v)iew > ").strip().lower()
                if cmd == 'h':
                    print(HELP); continue
                if cmd == 'q':
                    return
                if cmd == 'r':
                    try:
                        g.roll_dice(g.current_player)
                    except GameError as e:
                        print(f"Error: {e}")
                    show_all(g)
                    if g.current_player in bots and hasattr(bots[g.current_player], 'plan_turn'):
                        bots[g.current_player].plan_turn(g)
                    handle_white_phase(g, bots)
                    show_all(g)
                    if g.phase in (Phase.COLOR_DICE, Phase.COMPLETE):
                        if g.phase == Phase.COLOR_DICE:
                            handle_color_phase(g, bots)
                        try:
                            nxt, reason = g.end_turn(g.current_player)
                            print(f"Next player: {nxt}")
                            if reason:
                                print(f"Game ended: {reason}")
                        except GameError:
                            pass
                    show_all(g)
                elif cmd == 'v':
                    show_all(g)
                else:
                    continue

            elif g.phase == Phase.WHITE_SUM:
                if g.current_player in bots and getattr(bots[g.current_player], 'pending_plan', None) is None:
                    if hasattr(bots[g.current_player], 'plan_turn'):
                        bots[g.current_player].plan_turn(g)
                handle_white_phase(g, bots)
                show_all(g)
                if g.phase in (Phase.COLOR_DICE, Phase.COMPLETE):
                    if g.phase == Phase.COLOR_DICE:
                        handle_color_phase(g, bots)
                    try:
                        nxt, reason = g.end_turn(g.current_player)
                        print(f"Next player: {nxt}")
                        if reason:
                            print(f"Game ended: {reason}")
                    except GameError:
                        pass
                show_all(g)

            elif g.phase == Phase.COLOR_DICE:
                handle_color_phase(g, bots)
                show_all(g)
                try:
                    nxt, reason = g.end_turn(g.current_player)
                    print(f"Next player: {nxt}")
                    if reason:
                        print(f"Game ended: {reason}")
                except GameError:
                    pass
                show_all(g)

            elif g.phase == Phase.COMPLETE:
                try:
                    nxt, reason = g.end_turn(g.current_player)
                    print(f"Next player: {nxt}")
                    if reason:
                        print(f"Game ended: {reason}")
                except GameError as e:
                    print(f"Error: {e}")
                show_all(g)

        except KeyboardInterrupt:
            print('\nBye!')
            return
        except GameError as e:
            print(f"Error: {e}")

def main(argv: List[str]):
    parser = argparse.ArgumentParser(prog="qwixx.ui_cli", add_help=False)
    parser.add_argument("--bot", action="append", default=[], help="Name of a player to control with HeuristicBot (repeatable)")
    parser.add_argument("--mcts", action="append", default=[], help="Name of a player to control with MCTSBot (repeatable)")
    parser.add_argument("--learn", action="append", default=[], help="Name of a player to control with LearnerBot (repeatable)")
    parser.add_argument("--gap1", action="append", default=[], help="Name of a player to control with Gap1Bot (repeatable)")
    parser.add_argument("--gap1-lock", action="append", default=[], help="Name with Gap1LockBot (repeatable)")
    parser.add_argument("--gap2", action="append", default=[], help="Name with Gap2Bot (repeatable)")
    parser.add_argument("--gap2-lock", action="append", default=[], help="Name with Gap2LockBot (repeatable)")
    parser.add_argument("--gap3", action="append", default=[], help="Gap3 bot (max skip 3)")
    parser.add_argument("--gap3-lock", action="append", default=[], help="Gap3Lock bot (max skip 3, lock exception)")
    parser.add_argument("--probgap", action="append", default=[], help="Probability-aware gap bot (weighted skips)")
    parser.add_argument("--probgap-lock", action="append", default=[], help="Probability-aware gap bot with lock exception")
    parser.add_argument("--scout", action="append", default=[], help="Scout bot (hybrid heuristic + tiny MCTS)")
    parser.add_argument("--learn-file", type=str, default="./qwixx_data/learner.json", help="Learner store JSON path")
    parser.add_argument("--epsilon", type=float, default=0.10, help="Learner epsilon (exploration rate, initial)")
    parser.add_argument("--autoplay", type=int, default=0, help="Run N headless games and print report")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible autoplay series")
    parser.add_argument("--mcts-sims", type=int, default=400, help="Override MCTS simulations")
    parser.add_argument("--mcts-depth", type=int, default=2, help="Override MCTS rollout depth in turns")
    parser.add_argument("--show-each", action="store_true", help="Show per-game result lines in autoplay")
    parser.add_argument("--progress-interval", type=float, default=0.5, help="Seconds between progress updates (autoplay)")
    parser.add_argument("--debug-bots", action="store_true", help="Print player list and assigned bot types (autoplay)")
    parser.add_argument("--debug-timing", action="store_true", help="Print per-game elapsed time (autoplay)")
    parser.add_argument("--names", nargs="*", help="Player names in turn order")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Learner epsilon floor (decay target)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Per-game epsilon decay factor")

    args, _ = parser.parse_known_args(argv[1:])

    # --- Autoplay mode ---
    if args.autoplay > 0:
        if not args.names or len(args.names) < 2:
            print("Error: provide at least two player names for autoplay (e.g., L M)")
            return
        names = list(args.names)
        # Build bot-types; default heuristic
        bot_types: Dict[str, str] = {n: 'heur' for n in names}
        for n in (args.mcts or []):
            if n in bot_types: bot_types[n] = 'mcts'
        for n in (args.learn or []):
            if n in bot_types: bot_types[n] = 'learn'
        for n in (args.gap1 or []):
            if n in bot_types: bot_types[n] = 'gap1'
        for n in (args.gap1_lock or []):
            if n in bot_types: bot_types[n] = 'gap1lock'
        for n in (args.gap2 or []):
            if n in bot_types: bot_types[n] = 'gap2'
        for n in (args.gap2_lock or []):
            if n in bot_types: bot_types[n] = 'gap2lock'
        for n in (args.gap3 or []):
            if n in bot_types: bot_types[n] = 'gap3'
        for n in (args.gap3_lock or []):
            if n in bot_types: bot_types[n] = 'gap3lock'
        for n in (args.probgap or []):
            if n in bot_types: bot_types[n] = 'probgap'
        for n in (args.probgap_lock or []):
            if n in bot_types: bot_types[n] = 'probgaplock'
        for n in (args.scout or []):
            if n in bot_types: bot_types[n] = 'scout'


        if args.debug_bots:
            print(f"[DEBUG] Players: {names}")
            print(f"[DEBUG] bot_types: {bot_types}")

        report = play_series(
            names=names,
            bot_types=bot_types,
            games=args.autoplay,
            mcts_sims=args.mcts_sims,
            mcts_depth=args.mcts_depth,
            seed=args.seed,
            show_each=args.show_each,
            progress_interval=args.progress_interval,
            epsilon=args.epsilon,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            learn_file=args.learn_file,
            debug_timing=args.debug_timing
        )
        print("\n=== Autoplay Report ===")
        print(f"Games: {report['games']}")
        wins_sorted = {k: report['wins'].get(k, 0) for k in sorted(report['wins'].keys())}
        print(f"Wins : {wins_sorted} \nTies: {report['ties']}")
        print(f"Avg margin: {report['avg_margin']:.2f}")
        print("Avg scores:")
        for n in sorted(report['avg_scores'].keys()):
            print(f" - {n}: {report['avg_scores'][n]:.2f}")
        return

    # --- Interactive mode ---
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
        if n in names:
            bots[n] = HeuristicBot(n)
            print(f"Heuristic BOT enabled for player: {n}")
    for n in args.mcts or []:
        if n in names:
            bots[n] = MCTSBot(n, simulations=args.mcts_sims, depth_turns=args.mcts_depth)
            print(f"MCTS BOT enabled for player: {n}")
    for n in args.learn or []:
        if n in names:
            bots[n] = LearnerBot(n, epsilon=args.epsilon, store_path=args.learn_file)
            print(f"Learner BOT enabled for player: {n}")
    for n in args.gap1 or []:
        if n in names:
            bots[n] = Gap1Bot(n)
            print(f"Gap1 BOT (max skip 1) enabled for: {n}")
    for n in args.gap1_lock or []:
        if n in names:
            bots[n] = Gap1LockBot(n)
            print(f"Gap1Lock BOT (max skip 1, lock exception) enabled for: {n}")
    for n in args.gap2 or []:
        if n in names:
            bots[n] = Gap2Bot(n)
            print(f"Gap2 BOT (max skip 2) enabled for: {n}")
    for n in args.gap2_lock or []:
        if n in names:
            bots[n] = Gap2LockBot(n)
            print(f"Gap2Lock BOT (max skip 2, lock exception) enabled for: {n}")
    for n in args.gap3 or []:
        if n in names: 
            bots[n] = Gap3Bot(n)
    for n in args.gap3_lock or []:
        if n in names: 
            bots[n] = Gap3LockBot(n)
    for n in args.probgap or []:
        if n in names: 
            bots[n] = ProbGapBot(n)
    for n in args.probgap_lock or []:
        if n in names: 
            bots[n] = ProbGapLockBot(n)
    for n in args.scout or []:
        if n in names: 
            bots[n] = ScoutBot(n)
            

    prompt(g, bots)

if __name__ == '__main__':
    main(sys.argv)
