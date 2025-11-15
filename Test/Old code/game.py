# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import random

class Phase(str, Enum):
    INITIAL = 'initial'
    WHITE_SUM = 'whiteSum'
    COLOR_DICE = 'colorDice'
    COMPLETE = 'complete'

class GameStatus(str, Enum):
    WAITING = 'waiting'
    ACTIVE = 'active'
    ROLLING = 'rolling'
    FINISHED = 'finished'

COLORS = ['red','yellow','green','blue']
UP_ROWS = {'red', 'yellow'}      # 2 -> 12
DOWN_ROWS = {'green', 'blue'}    # 12 -> 2
LAST_NUMBER = {'red': 12, 'yellow': 12, 'green': 2, 'blue': 2}

class GameError(Exception):
    pass

@dataclass
class Player:
    name: str
    is_host: bool = False
    red: List[int] = field(default_factory=list)
    yellow: List[int] = field(default_factory=list)
    green: List[int] = field(default_factory=list)
    blue: List[int] = field(default_factory=list)
    penalties: int = 0

    def row(self, color: str) -> List[int]:
        return getattr(self, color)

    def to_scorecard_dict(self) -> Dict:
        return {
            'red': list(self.red),
            'yellow': list(self.yellow),
            'green': list(self.green),
            'blue': list(self.blue),
            'penalties': self.penalties,
        }

@dataclass
class QwixxGame:
    game_id: str
    players: List[Player]
    current_player: str
    status: GameStatus = GameStatus.WAITING
    phase: Phase = Phase.INITIAL
    initialized: bool = False
    dice: Dict[str, Optional[int]] = field(default_factory=lambda: {
        'white1': None,
        'white2': None,
        'red': None,
        'yellow': None,
        'green': None,
        'blue': None,
    })
    closed_rows: Dict[str, bool] = field(default_factory=lambda: {c: False for c in COLORS})
    active_dice: Dict[str, bool] = field(default_factory=lambda: {c: True for c in COLORS})
    row_closers: Dict[str, List[str]] = field(default_factory=lambda: {c: [] for c in COLORS})
    turn_actions: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    white_phase_complete: Dict[str, bool] = field(default_factory=dict)
    pending_row_closures: Dict[str, bool] = field(default_factory=lambda: {c: False for c in COLORS})
    selected_white_die: Optional[int] = None
    selected_color: Optional[str] = None

    def __post_init__(self):
        seen = set()
        for p in self.players:
            if p.name in seen:
                raise GameError(f"Duplicate player name: {p.name}")
            seen.add(p.name)
        if not any(p.is_host for p in self.players):
            self.players[0].is_host = True
        for p in self.players:
            self.turn_actions[p.name] = {
                'whiteSumMarked': False,
                'whiteSumSkipped': False,
                'colorDiceMarked': False,
            }
            self.white_phase_complete[p.name] = False

    def _player(self, name: str) -> Player:
        for p in self.players:
            if p.name == name:
                return p
        raise GameError(f"Unknown player: {name}")

    def is_active_player(self, name: str) -> bool:
        return self.current_player == name

    def next_player_name(self) -> str:
        idx = next((i for i, p in enumerate(self.players) if p.name == self.current_player), 0)
        return self.players[(idx + 1) % len(self.players)].name

    def white_sum(self) -> Optional[int]:
        w1, w2 = self.dice['white1'], self.dice['white2']
        if w1 is None or w2 is None:
            return None
        return w1 + w2

    def combined_value(self) -> Optional[int]:
        if self.selected_white_die in (1, 2) and self.selected_color in COLORS:
            wd = self.dice['white1'] if self.selected_white_die == 1 else self.dice['white2']
            cd = self.dice[self.selected_color]
            if wd is not None and cd is not None:
                return wd + cd
        return None

    def _validate_sequence(self, row: List[int], color: str, number: int) -> bool:
        if not row:
            return True
        if color in UP_ROWS:
            return number > max(row)
        if color in DOWN_ROWS:
            return number < min(row)
        return False

    def _sorted_insert(self, row: List[int], color: str, number: int) -> List[int]:
        if number in row:
            return row
        row2 = row + [number]
        if color in UP_ROWS:
            row2.sort()
        else:
            row2.sort(reverse=True)
        return row2

    def _count_closed_rows(self) -> int:
        return sum(1 for v in self.closed_rows.values() if v)

    def _check_end_conditions(self) -> Optional[str]:
        if self._count_closed_rows() >= 2:
            self.status = GameStatus.FINISHED
            return 'rowsClosed'
        for p in self.players:
            if p.penalties >= 4:
                self.status = GameStatus.FINISHED
                return 'penalties'
        return None

    def select_first_player(self, name: str):
        if name not in [p.name for p in self.players]:
            raise GameError('Invalid first player')
        self.current_player = name
        self.initialized = True
        self.status = GameStatus.ACTIVE
        self.phase = Phase.INITIAL

    def roll_dice(self, player: str):
        if not self.is_active_player(player):
            raise GameError('Only active player can roll dice')
        if not self.initialized:
            raise GameError('Game not initialized')
        if self.phase != Phase.INITIAL:
            raise GameError('Dice can be rolled only in the initial phase')
        self.status = GameStatus.ROLLING
        self.dice['white1'] = random.randint(1,6)
        self.dice['white2'] = random.randint(1,6)
        for c in COLORS:
            self.dice[c] = random.randint(1,6) if self.active_dice[c] else 0
        self.status = GameStatus.ACTIVE
        self.phase = Phase.WHITE_SUM
        for pn in self.turn_actions:
            self.turn_actions[pn] = {'whiteSumMarked': False, 'whiteSumSkipped': False, 'colorDiceMarked': False}
            self.white_phase_complete[pn] = False
        self.selected_white_die = None
        self.selected_color = None

    def mark_number_white_sum(self, player: str, color: str, number: int):
        if color not in COLORS:
            raise GameError('Invalid color')
        if self.phase != Phase.WHITE_SUM:
            raise GameError('Not in white sum phase')
        if self.turn_actions[player]['whiteSumMarked']:
            raise GameError('Already marked using white sum this turn')
        if self.turn_actions[player]['whiteSumSkipped']:
            raise GameError('You already skipped the white sum this turn')
        if self.closed_rows[color]:
            raise GameError(f'{color} row is closed')
        ws = self.white_sum()
        if ws is None or number != ws:
            raise GameError(f'White dice sum is {ws}, not {number}')
        pl = self._player(player)
        row = pl.row(color)
        if not self._validate_sequence(row, color, number):
            raise GameError('Sequence rule violated')
        setattr(pl, color, self._sorted_insert(row, color, number))
        self.turn_actions[player]['whiteSumMarked'] = True
        self.turn_actions[player]['whiteSumSkipped'] = False
        self.white_phase_complete[player] = True
        self._maybe_queue_row_closure(color, getattr(pl, color), action_type='whiteSum')

    def skip_white_sum(self, player: str):
        if self.phase != Phase.WHITE_SUM:
            raise GameError('Can only skip during white sum phase')
        self.turn_actions[player]['whiteSumSkipped'] = True
        self.turn_actions[player]['whiteSumMarked'] = False
        self.white_phase_complete[player] = True

    def all_white_phase_completed(self) -> bool:
        return all(self.white_phase_complete.get(p.name, False) for p in self.players)

    def proceed_to_color_phase_if_ready(self):
        if self.phase == Phase.WHITE_SUM and self.all_white_phase_completed():
            self.finalize_pending_row_closures()
            self.phase = Phase.COLOR_DICE

    def select_dice_for_color_phase(self, player: str, white_die: int, color: str):
        if not self.is_active_player(player):
            raise GameError('Only active player can select dice')
        if self.phase != Phase.COLOR_DICE:
            raise GameError('Not in color dice phase')
        if white_die not in (1,2):
            raise GameError('white_die must be 1 or 2')
        if color not in COLORS:
            raise GameError('Invalid color')
        if self.closed_rows[color]:
            raise GameError(f'{color} row is closed')
        if not self.active_dice[color]:
            raise GameError(f'{color} die is inactive')
        if self.turn_actions[player]['colorDiceMarked']:
            raise GameError('Already marked using color dice this turn')
        self.selected_white_die = white_die
        self.selected_color = color

    def mark_number_color_dice(self, player: str, color: str, number: int):
        if not self.is_active_player(player):
            raise GameError('Only active player can mark color dice')
        if self.phase != Phase.COLOR_DICE:
            raise GameError('Not in color dice phase')
        if self.turn_actions[player]['colorDiceMarked']:
            raise GameError('Already marked a color number this turn')
        if color not in COLORS:
            raise GameError('Invalid color')
        if self.selected_white_die not in (1,2) or self.selected_color != color:
            raise GameError('You must select a white die and the matching colored die first')
        wd = self.dice['white1'] if self.selected_white_die == 1 else self.dice['white2']
        cd = self.dice[color]
        if wd is None or cd is None:
            raise GameError('Dice not rolled')
        if number != wd + cd:
            raise GameError(f'Combined value is {wd+cd}, not {number}')
        if self.closed_rows[color]:
            raise GameError(f'{color} row is closed')
        pl = self._player(player)
        row = pl.row(color)
        if not self._validate_sequence(row, color, number):
            raise GameError('Sequence rule violated')
        setattr(pl, color, self._sorted_insert(row, color, number))
        self.turn_actions[player]['colorDiceMarked'] = True
        self._maybe_queue_row_closure(color, getattr(pl, color), action_type='colorDice')

    def take_penalty(self, player: str):
        if not self.is_active_player(player):
            raise GameError('Only active player can take penalties')
        pl = self._player(player)
        if pl.penalties >= 4:
            raise GameError('Already at max penalties')
        pl.penalties += 1
        self.phase = Phase.COMPLETE
        self.turn_actions[player]['whiteSumSkipped'] = True
        self.turn_actions[player]['whiteSumMarked'] = False
        self.turn_actions[player]['colorDiceMarked'] = False
        reason = self._check_end_conditions()
        return reason

    def end_turn(self, player: str) -> Tuple[str, Optional[str]]:
        if not self.is_active_player(player):
            raise GameError('Only active player can end their turn')
        if self.phase not in (Phase.COLOR_DICE, Phase.COMPLETE):
            raise GameError('Cannot end turn yet')
        ta = self.turn_actions[player]
        if self.phase == Phase.COLOR_DICE and ta['whiteSumSkipped'] and not ta['colorDiceMarked']:
            raise GameError('Must take a penalty when you skip white sum and do not mark color')
        self.selected_white_die = None
        self.selected_color = None
        nextp = self.next_player_name()
        self.current_player = nextp
        self.phase = Phase.INITIAL
        reason = self._check_end_conditions()
        if reason:
            return (nextp, reason)
        return (nextp, None)

    def _maybe_queue_row_closure(self, color: str, marked_numbers: List[int], action_type: str):
        last_num = LAST_NUMBER[color]
        if last_num in marked_numbers and len(marked_numbers) >= 6:
            if action_type == 'whiteSum':
                self.pending_row_closures[color] = True
            else:
                self._close_row_immediately(color, actor=self.current_player)

    def finalize_pending_row_closures(self):
        rows_to_close = [c for c, pending in self.pending_row_closures.items() if pending and not self.closed_rows[c]]
        if not rows_to_close:
            return
        for row in rows_to_close:
            self._close_row_immediately(row, actor=self.current_player)
        for row in rows_to_close:
            self.pending_row_closures[row] = False

    def _close_row_immediately(self, color: str, actor: str):
        if self.closed_rows[color]:
            return
        self.closed_rows[color] = True
        self.active_dice[color] = False
        if actor not in self.row_closers[color]:
            self.row_closers[color].append(actor)
        self._check_end_conditions()

    def calculate_score(self, player_name: str) -> Dict[str, int]:
        p = self._player(player_name)
        score = 0
        breakdown = {'red': 0, 'yellow': 0, 'green': 0, 'blue': 0, 'penalties': 0, 'total': 0}
        for color in COLORS:
            count = len(getattr(p, color))
            if count == 0:
                breakdown[color] = 0
                continue
            has_enough = count >= 6
            did_close = player_name in self.row_closers[color] and has_enough
            actual_count = count + 1 if did_close else count
            row_score = actual_count * (actual_count + 1) // 2
            breakdown[color] = row_score
            score += row_score
        penalty_points = p.penalties * 5
        score -= penalty_points
        breakdown['penalties'] = -penalty_points
        breakdown['total'] = score
        return breakdown

    def to_dict(self) -> Dict:
        return {
            'gameId': self.game_id,
            'players': [{'name': p.name, 'isHost': p.is_host} for p in self.players],
            'currentPlayer': self.current_player,
            'status': self.status.value,
            'phase': self.phase.value,
            'initialized': self.initialized,
            'dice': dict(self.dice),
            'closedRows': dict(self.closed_rows),
            'activeDice': dict(self.active_dice),
            'rowClosers': {k: list(v) for k, v in self.row_closers.items()},
            'turnActions': {k: dict(v) for k, v in self.turn_actions.items()},
            'whitePhaseComplete': dict(self.white_phase_complete),
            'pendingRowClosures': dict(self.pending_row_closures),
            'scorecards': {p.name: p.to_scorecard_dict() for p in self.players},
        }
