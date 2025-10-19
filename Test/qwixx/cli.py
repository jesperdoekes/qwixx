
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from typing import List
from .game import QwixxGame, Player, GameError, Phase, GameStatus

BOLD = '\033[1m'
RED = '\033[31m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'

def print_state(g: QwixxGame):
    d = g.to_dict()
    print(f"\n{BOLD}Game {g.game_id}{RESET} | status={d['status']} phase={d['phase']} current={d['currentPlayer']}")
    print(f"Dice: {d['dice']}")
    print(f"Closed rows: {d['closedRows']}  Active dice: {d['activeDice']}")
    for p in g.players:
        sc = p.to_scorecard_dict()
        print(f"- {BOLD}{p.name}{RESET} host={'*' if p.is_host else ' '}: R{sc['red']} Y{sc['yellow']} G{sc['green']} B{sc['blue']} | penalties={sc['penalties']}")
        bd = g.calculate_score(p.name)
        print(f"  -> score: red={bd['red']} yellow={bd['yellow']} green={bd['green']} blue={bd['blue']} penalties={bd['penalties']} total={bd['total']}")

def prompt(g: QwixxGame):
    print_state(g)
    cp = g.current_player
    while True:
        try:
            if g.status == GameStatus.FINISHED:
                print("Game finished. Thanks for playing!")
                return
            if g.phase == Phase.INITIAL:
                cmd = input(f"[{cp}] (r)oll dice / (q)uit > ").strip().lower()
                if cmd == 'r':
                    g.roll_dice(cp)
                elif cmd == 'q':
                    print('Bye!')
                    return
                else:
                    continue
            elif g.phase == Phase.WHITE_SUM:
                ws = g.white_sum()
                print(f"White-sum phase. Sum={ws}. Each player may mark the sum in any row.")
                sub = input(f"Action for (a)ll complete / (m)ark / (s)kip / (v)iew > ").strip().lower()
                if sub == 'm':
                    who = input('player name > ').strip()
                    color = input('color [red/yellow/green/blue] > ').strip()
                    try:
                        g.mark_number_white_sum(who, color, ws)
                        print(f"{who} marked {ws} in {color}")
                    except GameError as e:
                        print(f"Error: {e}")
                elif sub == 's':
                    who = input('player name > ').strip()
                    try:
                        g.skip_white_sum(who)
                        print(f"{who} skipped white sum")
                    except GameError as e:
                        print(f"Error: {e}")
                elif sub == 'a':
                    for p in g.players:
                        if not g.white_phase_complete[p.name]:
                            g.skip_white_sum(p.name)
                    print('All players completed white phase (remaining were skipped).')
                elif sub == 'v':
                    print_state(g)
                else:
                    pass
                g.proceed_to_color_phase_if_ready()
            elif g.phase == Phase.COLOR_DICE:
                print('Color-dice phase (active player only).')
                print(f"Selected: white={g.selected_white_die} color={g.selected_color}")
                sub = input(f"[{cp}] (s)elect dice / (m)ark / (p)enalty / (e)nd turn / (v)iew > ").strip().lower()
                if sub == 's':
                    try:
                        wd = int(input('Choose white die [1/2] > ').strip())
                        col = input('Choose color [red/yellow/green/blue] > ').strip()
                        g.select_dice_for_color_phase(cp, wd, col)
                    except Exception as e:
                        print(f"Error: {e}")
                elif sub == 'm':
                    col = g.selected_color or input('color [red/yellow/green/blue] > ').strip()
                    try:
                        cv = g.combined_value()
                        if cv is None:
                            print('Select dice first.')
                            continue
                        g.mark_number_color_dice(cp, col, cv)
                        print(f"Marked {cv} in {col}")
                    except GameError as e:
                        print(f"Error: {e}")
                elif sub == 'p':
                    try:
                        reason = g.take_penalty(cp)
                        if reason:
                            print(f"Game end reason: {reason}")
                    except GameError as e:
                        print(f"Error: {e}")
                elif sub == 'e':
                    try:
                        nxt, reason = g.end_turn(cp)
                        print(f"Next player: {nxt}")
                        if reason:
                            print(f"Game end reason: {reason}")
                        cp = g.current_player
                    except GameError as e:
                        print(f"Error: {e}")
                elif sub == 'v':
                    print_state(g)
                else:
                    pass
            elif g.phase == Phase.COMPLETE:
                sub = input(f"[{cp}] Turn complete. (e)nd turn > ").strip().lower()
                if sub == 'e':
                    try:
                        nxt, reason = g.end_turn(cp)
                        print(f"Next player: {nxt}")
                        if reason:
                            print(f"Game end reason: {reason}")
                        cp = g.current_player
                    except GameError as e:
                        print(f"Error: {e}")
                else:
                    pass
        except KeyboardInterrupt:
            print('\nBye!')
            return
        except GameError as e:
            print(f"Error: {e}")

def main(argv: List[str]):
    print(f"Qwixx (CLI) - Python port. Ctrl+C to exit.")
    game_id = 'local'
    if len(argv) >= 2:
        names = argv[1:]
    else:
        n1 = input('Player 1 name > ').strip() or 'Alice'
        n2 = input('Player 2 name > ').strip() or 'Bob'
        names = [n1, n2]
    players = [Player(n, is_host=(i==0)) for i, n in enumerate(names)]
    g = QwixxGame(game_id=game_id, players=players, current_player=players[0].name)
    g.select_first_player(players[0].name)
    prompt(g)

if __name__ == '__main__':
    main(sys.argv)
