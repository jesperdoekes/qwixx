# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from .game import QwixxGame, Player, COLORS

# ANSI colors
BOLD = '\033[1m'
DIM = '\033[2m'
RED = '\033[31m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
BLUE = '\033[34m'
GREY = '\033[90m'
RESET = '\033[0m'

COLOR_SHORT = {'red':'R','yellow':'Y','green':'G','blue':'B'}
SHORT_TO_COLOR = {v:k for k,v in COLOR_SHORT.items()}
COLOR_ANSI = {'red':RED, 'yellow':YELLOW, 'green':GREEN, 'blue':BLUE}

ROW_NUMBERS = {
    'red': list(range(2,13)),
    'yellow':list(range(2,13)),
    'green': list(range(12,1,-1)),
    'blue': list(range(12,1,-1)),
}

def number_index(color: str, number: int) -> Optional[int]:
    if number < 2 or number > 12:
        return None
    if color in ('red','yellow'):
        return number - 2
    else:
        return 12 - number

def sequence_valid(marked: List[int], color: str, number: int) -> bool:
    if not marked:
        return True
    if color in ('red','yellow'):
        return number > max(marked)
    return number < min(marked)

def render_player_sheet(g: QwixxGame, player: Player) -> str:
    lines: List[str] = []
    for color in COLORS:
        nums = ROW_NUMBERS[color]
        marks = []
        row = getattr(player, color)
        ansi = COLOR_ANSI[color]
        for n in nums:
            mk = 'X' if n in row else '.'
            marks.append(f"{n:2}:{mk}")
        locks = ' 🔔' if g.closed_rows[color] else ''
        lines.append(f"{ansi}{COLOR_SHORT[color]}{RESET}: " + ' '.join(marks) + locks)
    lines.append(f"Penalties: {player.penalties}")
    return '\n'.join(lines)

def render_score(g: QwixxGame, player: Player) -> str:
    bd = g.calculate_score(player.name)
    return (
        f"Score ▶ R={bd['red']} Y={bd['yellow']} G={bd['green']} B={bd['blue']} "
        f"pen={bd['penalties']} → {BOLD}{bd['total']}{RESET}"
    )

def render_roll(g: QwixxGame) -> str:
    d = g.dice
    wsum = (d['white1'] or 0) + (d['white2'] or 0) if (d['white1'] and d['white2']) else None
    lines = [
        f"White dice: [1]={d['white1']} [2]={d['white2']} sum={wsum}",
        "Colored dice:"
    ]
    for color in COLORS:
        if not g.active_dice[color]:
            lines.append(f" {COLOR_SHORT[color]}: (inactive)")
            continue
        cval = d[color]
        lines.append(
            f" {COLOR_SHORT[color]}: {cval} (w1+{COLOR_SHORT[color]}={ (d['white1'] or 0) + (cval or 0) }"
            f", w2+{COLOR_SHORT[color]}={ (d['white2'] or 0) + (cval or 0) })"
        )
    return '\n'.join(lines)

def suggest_white_sum_targets(g: QwixxGame, player: Player) -> List[str]:
    d = g.dice
    if d['white1'] is None or d['white2'] is None:
        return []
    ws = d['white1'] + d['white2']
    options: List[str] = []
    for color in COLORS:
        if g.closed_rows[color]:
            continue
        row = getattr(player, color)
        if ws in ROW_NUMBERS[color] and (ws not in row) and sequence_valid(row, color, ws):
            options.append(COLOR_SHORT[color])
    return options

def render_color_combos(g: QwixxGame, player: Player) -> str:
    d = g.dice
    if d['white1'] is None or d['white2'] is None:
        return "(dice not rolled)"
    lines = ["Combos (valid *):"]
    for color in COLORS:
        label = COLOR_SHORT[color]
        if g.closed_rows[color] or not g.active_dice[color]:
            lines.append(f" {label}: inactive/closed")
            continue
        row = getattr(player, color)
        w1 = d['white1'] + d[color]
        w2 = d['white2'] + d[color]
        v1 = '*' if (w1 not in row and sequence_valid(row, color, w1)) else ''
        v2 = '*' if (w2 not in row and sequence_valid(row, color, w2)) else ''
        lines.append(f" {label}: w1={w1}{v1} w2={w2}{v2}")
    return '\n'.join(lines)
