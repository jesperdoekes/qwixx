# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
import copy, random

from .game import QwixxGame, Player, COLORS, GameError
from .render import sequence_valid, ROW_NUMBERS
from .learn import ExperienceStore, FeatureHasher, plan_key, default_reward_from_scores

__all__ = [
    # Core bots
    'HeuristicBot', 'LearnerBot',
    # Gap-policy bots
    'GapPolicyBot', 'Gap1Bot', 'Gap1LockBot', 'Gap2Bot', 'Gap2LockBot',
    'Gap3Bot', 'Gap3LockBot',
    # Probability-aware gaps
    'ProbGapBot', 'ProbGapLockBot',
    # Search bots
    'MCTSBot', 'ScoutBot',
    # Shared legal action helpers
    'legal_white_sum_colors', 'legal_color_actions'
]

# ==================== Heuristic constants ====================
CLOSE_NOW_BONUS = 9.0
SETUP_BONUS = 4.0
FLEX_WEIGHT = 0.15
BIG_SKIP_PENALTY_PER_CELL = 1.25
NO_LOCK_CAP_PENALTY = 6.0
MIN_COMPONENT_VALUE = 0.75
COMBO_DELTA_MARGIN = 0.25

# ==================== MCTS defaults ====================
MCTS_SIMULATIONS = 400
MCTS_DEPTH_TURNS = 2
MCTS_BEAM = 10

# ==================== Shared helpers ====================
def score_total(g: QwixxGame, name: str) -> int:
    return g.calculate_score(name)['total']

def clone(g: QwixxGame) -> QwixxGame:
    return copy.deepcopy(g)

def last_number(color: str) -> int:
    return 12 if color in ('red','yellow') else 2

def frontier(row: List[int], color: str) -> Optional[int]:
    if not row:
        return None
    return max(row) if color in ('red','yellow') else min(row)

def sorted_insert_preview(row: List[int], color: str, number: int) -> List[int]:
    """Preview the row after inserting number (no duplicates)."""
    if number in row:
        return list(row)
    r2 = list(row) + [number]
    if color in ('red','yellow'):
        r2.sort()
    else:
        r2.sort(reverse=True)
    return r2

def big_skip_cells(row: List[int], color: str, new_num: int) -> int:
    f = frontier(row, color)
    if f is None:
        return 0
    if color in ('red','yellow'):
        return max(0, new_num - (f + 1))
    else:
        return max(0, (f - 1) - new_num)

def skip_from_start(row: List[int], color: str, new_num: int) -> int:
    """If empty row, measure skip from start (2 for R/Y; 12 for G/B). Else from frontier."""
    if row:
        return big_skip_cells(row, color, new_num)
    return max(0, new_num - 2) if color in ('red','yellow') else max(0, 12 - new_num)

def max_possible_marks_after(row_after: List[int], color: str) -> int:
    """Upper bound on how many cells can be marked in the row after this placement."""
    n = len(row_after)
    if not row_after:
        return 11
    if color in ('red','yellow'):
        hi = max(row_after)
        future = max(0, 12 - hi)
    else:
        lo = min(row_after)
        future = max(0, lo - 2)
    return n + future

def infeasible_to_lock(row_after: List[int], color: str) -> bool:
    """True if it's impossible to ever reach 6 marks from this state."""
    return max_possible_marks_after(row_after, color) < 6

def lock_feasible_after(row_before: List[int], color: str, number: int) -> bool:
    """Calling this before marking ensures that locking 6+ remains possible."""
    return not infeasible_to_lock(sorted_insert_preview(row_before, color, number), color)

def is_last_pick(color: str, number: int) -> bool:
    return number == last_number(color)

def last_pick_allowed(row_before: List[int], color: str, number: int) -> bool:
    """Must have ≥5 marks before taking the last number."""
    return (not is_last_pick(color, number)) or (len(row_before) >= 5)

def flexibility_score(row_after: List[int], color: str) -> int:
    if not row_after:
        return 11
    return max(0, (12 - max(row_after)) if color in ('red','yellow') else (min(row_after) - 2))

def will_close_now(g_after: QwixxGame, color: str, actor: str) -> bool:
    return g_after.closed_rows[color] and (actor in g_after.row_closers[color])

# ==================== Global legal action filters (apply to ALL bots) ====================
def legal_white_sum_colors(g: QwixxGame, name: str) -> List[str]:
    """Colors where the white-sum is playable for player 'name', enforcing:
       - open row, number in row, not taken,
       - sequence valid,
       - ≥5-before-last rule,
       - lock-feasibility after the mark.
    """
    ws = g.white_sum()
    if ws is None or g.phase != g.phase.WHITE_SUM:
        return []
    pl = next(p for p in g.players if p.name == name)
    ok: List[str] = []
    for color in COLORS:
        if g.closed_rows[color]:
            continue
        row = getattr(pl, color)
        if ws not in ROW_NUMBERS[color] or ws in row:
            continue
        if not sequence_valid(row, color, ws):
            continue
        if not last_pick_allowed(row, color, ws):
            continue
        if not lock_feasible_after(row, color, ws):
            continue
        ok.append(color)
    return ok

def legal_color_actions(g: QwixxGame, name: str) -> List[Tuple[str,int,int]]:
    """Returns (color, white_die_index, number) options for the active player's color phase,
       enforcing the same policies as above."""
    if g.phase != g.phase.COLOR_DICE:
        return []
    pl = next(p for p in g.players if p.name == name)
    d = g.dice
    if d['white1'] is None or d['white2'] is None:
        return []
    actions: List[Tuple[str,int,int]] = []
    for color in COLORS:
        if g.closed_rows[color] or not g.active_dice[color]:
            continue
        row = getattr(pl, color)
        # with white1
        n1 = d['white1'] + d[color]
        if (n1 not in row) and sequence_valid(row, color, n1) \
           and last_pick_allowed(row, color, n1) \
           and lock_feasible_after(row, color, n1):
            actions.append((color, 1, n1))
        # with white2
        n2 = d['white2'] + d[color]
        if (n2 not in row) and sequence_valid(row, color, n2) \
           and last_pick_allowed(row, color, n2) \
           and lock_feasible_after(row, color, n2):
            actions.append((color, 2, n2))
    return actions

# ==================== HeuristicBot ====================
@dataclass
class HeuristicBot:
    name: str
    pending_plan: Optional[Dict] = field(default=None, repr=False)

    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        if self.pending_plan and self.pending_plan.get('white_action'):
            return self.pending_plan['white_action']
        options = legal_white_sum_colors(g, self.name)
        if not options:
            return ('skip', None)
        base = score_total(g, self.name)
        best, best_val = ('skip', None), -1e9
        ws = g.white_sum()
        for color in options:
            sim = clone(g)
            try:
                row_before = getattr(next(p for p in g.players if p.name == self.name), color)
                sim.mark_number_white_sum(self.name, color, ws)
                sim.finalize_pending_row_closures()
                delta = score_total(sim, self.name) - base
                row_after = getattr(next(p for p in sim.players if p.name == self.name), color)
                skipped = big_skip_cells(row_before, color, ws)
                val = delta
                val -= BIG_SKIP_PENALTY_PER_CELL * skipped
                if will_close_now(sim, color, self.name): val += CLOSE_NOW_BONUS
                # soft setup: near the end, no skip
                if len(row_after) >= 4 and (last_number(color) not in row_after) and skipped == 0:
                    val += SETUP_BONUS
                if infeasible_to_lock(row_after, color): val -= NO_LOCK_CAP_PENALTY
                val += FLEX_WEIGHT * flexibility_score(row_after, color)
                if val > best_val:
                    best_val, best = val, ('mark', color)
            except GameError:
                continue
        return best

    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan and self.pending_plan.get('color_action'):
            return self.pending_plan['color_action']
        marked_white = bool(g.turn_actions.get(self.name, {}).get('whiteSumMarked', False))
        actions = legal_color_actions(g, self.name)
        if not actions:
            return ('skip', None, None)
        base = score_total(g, self.name)
        best, best_val = ('skip', None, None), -1e9
        for color, wd, number in actions:
            sim = clone(g)
            try:
                row_before = getattr(next(p for p in g.players if p.name == self.name), color)
                sim.select_dice_for_color_phase(self.name, wd, color)
                sim.mark_number_color_dice(self.name, color, number)
                delta = score_total(sim, self.name) - base
                row_after = getattr(next(p for p in sim.players if p.name == self.name), color)
                skipped = big_skip_cells(row_before, color, number)
                val = delta + (0.5 if not marked_white else 0.0)
                val -= BIG_SKIP_PENALTY_PER_CELL * skipped
                if will_close_now(sim, color, self.name): val += CLOSE_NOW_BONUS
                if len(row_after) >= 4 and (last_number(color) not in row_after) and skipped == 0:
                    val += SETUP_BONUS
                if infeasible_to_lock(row_after, color): val -= NO_LOCK_CAP_PENALTY
                val += FLEX_WEIGHT * flexibility_score(row_after, color)
                if val > best_val:
                    best_val, best = val, ('mark', color, wd)
            except GameError:
                continue
        return best

    def plan_turn(self, g: QwixxGame):
        base = score_total(g, self.name)
        ws = g.white_sum()
        white_choices: List[Tuple[str, Optional[str]]] = [('skip', None)]
        white_choices += [('mark', c) for c in legal_white_sum_colors(g, self.name)]
        best_pair = None
        best_total_val = -1e9
        best_white_only_val = -1e9
        best_color_only_val = -1e9
        best_white_only: Tuple[str, Optional[str]] = ('skip', None)
        best_color_only: Tuple[str, Optional[str], Optional[int]] = ('skip', None, None)

        def w_value(row_before, color_w, sim_afterW, delta_white) -> float:
            row_after = getattr(next(p for p in sim_afterW.players if p.name == self.name), color_w)
            skipped = big_skip_cells(row_before, color_w, ws)
            val = delta_white
            val -= BIG_SKIP_PENALTY_PER_CELL * skipped
            if will_close_now(sim_afterW, color_w, self.name): val += CLOSE_NOW_BONUS
            if len(row_after) >= 4 and (last_number(color_w) not in row_after) and skipped == 0:
                val += SETUP_BONUS
            if infeasible_to_lock(row_after, color_w): val -= NO_LOCK_CAP_PENALTY
            val += FLEX_WEIGHT * flexibility_score(row_after, color_w)
            return val

        def c_value(row_before, color_c, wd, number, sim_afterC, delta_color) -> float:
            row_after = getattr(next(p for p in sim_afterC.players if p.name == self.name), color_c)
            skipped = big_skip_cells(row_before, color_c, number)
            val = delta_color
            val -= BIG_SKIP_PENALTY_PER_CELL * skipped
            if will_close_now(sim_afterC, color_c, self.name): val += CLOSE_NOW_BONUS
            if len(row_after) >= 4 and (last_number(color_c) not in row_after) and skipped == 0:
                val += SETUP_BONUS
            if infeasible_to_lock(row_after, color_c): val -= NO_LOCK_CAP_PENALTY
            val += FLEX_WEIGHT * flexibility_score(row_after, color_c)
            return val

        for w_action, w_color in white_choices:
            simW = clone(g)
            white_marked = False
            delta_white = 0.0
            w_component_val = 0.0
            if w_action == 'mark' and w_color and ws is not None:
                row_before_w = getattr(next(p for p in g.players if p.name == self.name), w_color)
                try:
                    simW.mark_number_white_sum(self.name, w_color, ws)
                    simW.finalize_pending_row_closures()
                    white_marked = True
                    delta_white = score_total(simW, self.name) - base
                    w_component_val = w_value(row_before_w, w_color, simW, delta_white)
                except GameError:
                    white_marked = False
                    delta_white = 0.0
                    w_component_val = 0.0
            else:
                try: simW.skip_white_sum(self.name)
                except GameError: pass
            simW.phase = simW.phase.COLOR_DICE

            acts = legal_color_actions(simW, self.name)
            color_choices: List[Tuple[str, Optional[str], Optional[int]]] = [('skip', None, None)]
            color_choices += [('mark', c, wd) for (c, wd, _) in acts for _ in [(0)]]

            if not white_marked:
                for (act_c, color_c, wd_c) in color_choices:
                    if act_c == 'skip': continue
                    try:
                        simC = clone(simW)
                        d = simC.dice
                        number = (d['white1'] + d[color_c]) if wd_c == 1 else (d['white2'] + d[color_c])
                        row_before_c = getattr(next(p for p in simW.players if p.name == self.name), color_c)
                        simC.select_dice_for_color_phase(self.name, wd_c, color_c)
                        simC.mark_number_color_dice(self.name, color_c, number)
                        delta_color_only = score_total(simC, self.name) - base
                        c_val_only = c_value(row_before_c, color_c, wd_c, number, simC, delta_color_only)
                        if c_val_only > best_color_only_val:
                            best_color_only_val = c_val_only
                            best_color_only = ('mark', color_c, wd_c)
                    except GameError:
                        continue

            if white_marked and w_component_val > best_white_only_val:
                best_white_only_val = w_component_val
                best_white_only = ('mark', w_color)

            for (act_c, color_c, wd_c) in color_choices:
                if (not white_marked) and (act_c == 'skip'):
                    continue
                simPair = clone(simW)
                delta_color = 0.0
                c_component_val = 0.0
                if act_c == 'mark' and color_c and wd_c:
                    try:
                        d = simPair.dice
                        number = (d['white1'] + d[color_c]) if wd_c == 1 else (d['white2'] + d[color_c])
                        row_before_c = getattr(next(p for p in simW.players if p.name == self.name), color_c)
                        simPair.select_dice_for_color_phase(self.name, wd_c, color_c)
                        simPair.mark_number_color_dice(self.name, color_c, number)
                        delta_color = score_total(simPair, self.name) - (base + delta_white)
                        c_component_val = c_value(row_before_c, color_c, wd_c, number, simPair, delta_color)
                    except GameError:
                        continue
                total_val = w_component_val + c_component_val
                if total_val > best_total_val:
                    best_total_val = total_val
                    best_pair = {
                        'white_action': ('mark', w_color) if white_marked else ('skip', None),
                        'color_action': ('mark', color_c, wd_c) if act_c == 'mark' else ('skip', None, None),
                        'white_component_val': w_component_val,
                        'color_component_val': c_component_val,
                        'total_val': total_val,
                    }

        if best_pair is None:
            if best_color_only_val >= best_white_only_val and best_color_only_val > -1e9:
                self.pending_plan = {'white_action': ('skip', None), 'color_action': best_color_only}
            elif best_white_only_val > -1e9:
                self.pending_plan = {'white_action': best_white_only, 'color_action': ('skip', None, None)}
            else:
                self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None)}
            return

        plan_w, plan_c = best_pair['white_action'], best_pair['color_action']
        w_comp = best_pair['white_component_val']
        c_comp = best_pair['color_component_val']
        total = best_pair['total_val']

        if plan_w[0] == 'mark' and plan_c[0] == 'mark':
            best_single_val = max(best_white_only_val, best_color_only_val)
            if w_comp < MIN_COMPONENT_VALUE and (total - best_single_val) < COMBO_DELTA_MARGIN:
                plan_w = ('skip', None); total = c_comp
            elif c_comp < MIN_COMPONENT_VALUE and (total - best_single_val) < COMBO_DELTA_MARGIN:
                plan_c = ('skip', None, None); total = w_comp

        self.pending_plan = {'white_action': plan_w, 'color_action': plan_c, 'value': total}

# ==================== Gap-policy bots ====================
@dataclass
class GapPolicyBot(HeuristicBot):
    """Heuristic + policy: limit raw skip length. May break the cap only to lock now."""
    max_skip: int = 1
    allow_break_for_lock: bool = False

    def _would_lock(self, row_before: List[int], color: str, number: int) -> bool:
        return is_last_pick(color, number) and (len(row_before) + 1 >= 6)

    def _allowed_by_skip(self, row_before: List[int], color: str, number: int) -> bool:
        raw = skip_from_start(row_before, color, number)
        if raw <= self.max_skip:
            return True
        return self.allow_break_for_lock and self._would_lock(row_before, color, number)

    def _filtered_white_options(self, g: QwixxGame) -> List[str]:
        ws = g.white_sum()
        opts = legal_white_sum_colors(g, self.name)
        out: List[str] = []
        for color in opts:
            row = getattr(next(p for p in g.players if p.name == self.name), color)
            if self._allowed_by_skip(row, color, ws):
                out.append(color)
        return out

    def _filtered_color_actions(self, g: QwixxGame) -> List[Tuple[str,int,int]]:
        acts = legal_color_actions(g, self.name)
        out: List[Tuple[str,int,int]] = []
        for color, wd, num in acts:
            row = getattr(next(p for p in g.players if p.name == self.name), color)
            if self._allowed_by_skip(row, color, num):
                out.append((color, wd, num))
        return out

    # Use Heuristic evaluation over the filtered sets
    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        if self.pending_plan and self.pending_plan.get('white_action'):
            return self.pending_plan['white_action']
        options = self._filtered_white_options(g)
        if not options: return ('skip', None)
        base = score_total(g, self.name)
        best, best_val = ('skip', None), -1e9
        ws = g.white_sum()
        for color in options:
            sim = clone(g)
            try:
                row_before = getattr(next(p for p in g.players if p.name == self.name), color)
                sim.mark_number_white_sum(self.name, color, ws)
                sim.finalize_pending_row_closures()
                delta = score_total(sim, self.name) - base
                row_after = getattr(next(p for p in sim.players if p.name == self.name), color)
                skipped = big_skip_cells(row_before, color, ws)
                val = delta - BIG_SKIP_PENALTY_PER_CELL * skipped
                if will_close_now(sim, color, self.name): val += CLOSE_NOW_BONUS
                if len(row_after) >= 4 and (last_number(color) not in row_after) and skipped == 0:
                    val += SETUP_BONUS
                if infeasible_to_lock(row_after, color): val -= NO_LOCK_CAP_PENALTY
                val += FLEX_WEIGHT * flexibility_score(row_after, color)
                if val > best_val:
                    best_val, best = val, ('mark', color)
            except GameError:
                continue
        return best

    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan and self.pending_plan.get('color_action'):
            return self.pending_plan['color_action']
        marked_white = bool(g.turn_actions.get(self.name, {}).get('whiteSumMarked', False))
        actions = self._filtered_color_actions(g)
        if not actions: return ('skip', None, None)
        base = score_total(g, self.name)
        best, best_val = ('skip', None, None), -1e9
        for color, wd, num in actions:
            sim = clone(g)
            try:
                row_before = getattr(next(p for p in g.players if p.name == self.name), color)
                sim.select_dice_for_color_phase(self.name, wd, color)
                sim.mark_number_color_dice(self.name, color, num)
                delta = score_total(sim, self.name) - base
                row_after = getattr(next(p for p in sim.players if p.name == self.name), color)
                skipped = big_skip_cells(row_before, color, num)
                val = delta + (0.5 if not marked_white else 0.0)
                val -= BIG_SKIP_PENALTY_PER_CELL * skipped
                if will_close_now(sim, color, self.name): val += CLOSE_NOW_BONUS
                if len(row_after) >= 4 and (last_number(color) not in row_after) and skipped == 0:
                    val += SETUP_BONUS
                if infeasible_to_lock(row_after, color): val -= NO_LOCK_CAP_PENALTY
                val += FLEX_WEIGHT * flexibility_score(row_after, color)
                if val > best_val:
                    best_val, best = val, ('mark', color, wd)
            except GameError:
                continue
        return best

class Gap1Bot(GapPolicyBot):
    def __init__(self, name: str): super().__init__(name=name, max_skip=1, allow_break_for_lock=False)

class Gap1LockBot(GapPolicyBot):
    def __init__(self, name: str): super().__init__(name=name, max_skip=1, allow_break_for_lock=True)

class Gap2Bot(GapPolicyBot):
    def __init__(self, name: str): super().__init__(name=name, max_skip=2, allow_break_for_lock=False)

class Gap2LockBot(GapPolicyBot):
    def __init__(self, name: str): super().__init__(name=name, max_skip=2, allow_break_for_lock=True)

class Gap3Bot(GapPolicyBot):
    def __init__(self, name: str): super().__init__(name=name, max_skip=3, allow_break_for_lock=False)

class Gap3LockBot(GapPolicyBot):
    def __init__(self, name: str): super().__init__(name=name, max_skip=3, allow_break_for_lock=True)

# ==================== Probability-aware gap bots ====================
_2D6_COUNTS = {2:1,3:2,4:3,5:4,6:5,7:6,8:5,9:4,10:3,11:2,12:1}
_PEAK = _2D6_COUNTS[7]
_D6NORM = {n: _2D6_COUNTS[n] / _PEAK for n in range(2,13)}  # 7 → 1.0; 12/2 → 0.17

def _skipped_integers(row: List[int], color: str, new_num: int) -> Iterable[int]:
    """Integers strictly between the current frontier (or start) and new_num."""
    if color in ('red','yellow'):
        start = (max(row) if row else 2)
        lo, hi = start, new_num
        if hi <= lo + 1: return []
        return range(lo+1, hi)
    else:
        start = (min(row) if row else 12)
        lo, hi = new_num, start
        if hi <= lo + 1: return []
        return range(lo+1, hi)

@dataclass
class ProbGapBot(GapPolicyBot):
    """Gap policy with a probability-weighted skip budget (skipping near 7 is 'expensive')."""
    max_eff_skip: float = 1.75  # ~two cheap extremes, but not across 7

    def _effective_skip_weight(self, row_before: List[int], color: str, number: int) -> float:
        skipped = list(_skipped_integers(row_before, color, number))
        return sum(_D6NORM.get(n, 1.0) for n in skipped)

    def _allowed_by_skip(self, row_before: List[int], color: str, number: int) -> bool:
        if not super()._allowed_by_skip(row_before, color, number):
            return False
        w = self._effective_skip_weight(row_before, color, number)
        if w <= self.max_eff_skip:
            return True
        return self.allow_break_for_lock and self._would_lock(row_before, color, number)

class ProbGapLockBot(ProbGapBot):
    def __init__(self, name: str, max_eff_skip: float = 1.75):
        super().__init__(name=name, max_skip=2, allow_break_for_lock=True, max_eff_skip=max_eff_skip)

# ==================== MCTSBot (improved: priors + beam) ====================
def _plan_heuristic_prior(g: QwixxGame, me: str,
                          w_act: Tuple[str, Optional[str]],
                          c_act: Tuple[str, Optional[str], Optional[int]]) -> float:
    """Immediate one-turn delta as a cheap prior; -1e9 on illegal."""
    base = score_total(g, me)
    sim = clone(g)
    try:
        # white
        if w_act[0] == 'mark' and w_act[1]:
            ws = sim.white_sum()
            sim.mark_number_white_sum(me, w_act[1], ws)
            sim.finalize_pending_row_closures()
        else:
            sim.skip_white_sum(me)
        # move to color
        if sim.phase == sim.phase.WHITE_SUM:
            for p in sim.players:
                if not sim.white_phase_complete.get(p.name, False):
                    try: sim.skip_white_sum(p.name)
                    except GameError: pass
            sim.proceed_to_color_phase_if_ready()
        # color
        if c_act[0] == 'mark' and c_act[1] and c_act[2]:
            d = sim.dice
            number = (d['white1'] + d[c_act[1]]) if c_act[2] == 1 else (d['white2'] + d[c_act[1]])
            sim.select_dice_for_color_phase(me, c_act[2], c_act[1])
            sim.mark_number_color_dice(me, c_act[1], number)
        return score_total(sim, me) - base
    except GameError:
        return -1e9

@dataclass
class MCTSBot:
    name: str
    simulations: int = MCTS_SIMULATIONS
    depth_turns: int = MCTS_DEPTH_TURNS
    beam_width: int = MCTS_BEAM
    rollout_kind: str = 'heur'   # 'heur' | 'gap2' | 'gap2lock' | 'probgap'
    pending_plan: Optional[Dict] = field(default=None, repr=False)

    def _rollout_policy(self, pname: str) -> HeuristicBot:
        if self.rollout_kind == 'gap2':
            return Gap2Bot(pname)
        if self.rollout_kind == 'gap2lock':
            return Gap2LockBot(pname)
        if self.rollout_kind == 'probgap':
            return ProbGapBot(pname)
        return HeuristicBot(pname)

    def _apply_white(self, sim: QwixxGame, who: str, action: Tuple[str, Optional[str]]):
        act, color = action
        if act == 'mark' and color:
            ws = sim.white_sum()
            if ws is None: return
            sim.mark_number_white_sum(who, color, ws)
            sim.finalize_pending_row_closures()
        else:
            sim.skip_white_sum(who)

    def _apply_color(self, sim: QwixxGame, who: str, action: Tuple[str, Optional[str], Optional[int]]):
        act, color, wd = action
        marked_white = bool(sim.turn_actions.get(who, {}).get('whiteSumMarked', False))
        if act == 'mark' and color and wd:
            d = sim.dice
            number = (d['white1'] + d[color]) if wd == 1 else (d['white2'] + d[color])
            sim.select_dice_for_color_phase(who, wd, color)
            sim.mark_number_color_dice(who, color, number)
        else:
            if not marked_white:
                sim.take_penalty(who)

    def _simulate_future(self, g: QwixxGame, myname: str, depth_turns: int):
        bots: Dict[str, HeuristicBot] = {p.name: self._rollout_policy(p.name) for p in g.players}
        for _ in range(depth_turns):
            if g.status == g.status.FINISHED: break
            active = g.current_player
            try: g.roll_dice(active)
            except GameError: return score_total(g, myname)

            # white for all
            for p in g.players:
                if g.white_phase_complete.get(p.name, False): continue
                bot = bots[p.name]
                if p.name == active:
                    if getattr(bot, 'pending_plan', None) is None and hasattr(bot, 'plan_turn'): bot.plan_turn(g)
                    w_act = bot.pending_plan.get('white_action', ('skip', None)) if getattr(bot, 'pending_plan', None) else bot.choose_white_sum_action(g)
                else:
                    w_act = bot.choose_white_sum_action(g)
                try: self._apply_white(g, p.name, w_act)
                except GameError:
                    try: g.skip_white_sum(p.name)
                    except GameError: pass

            g.proceed_to_color_phase_if_ready()
            if g.phase == g.phase.COLOR_DICE:
                bot = bots[active]
                if getattr(bot, 'pending_plan', None) is None and hasattr(bot, 'plan_turn'): bot.plan_turn(g)
                c_act = bot.pending_plan.get('color_action', ('skip', None, None)) if getattr(bot, 'pending_plan', None) else bot.choose_color_action(g)
                try: self._apply_color(g, active, c_act)
                except GameError:
                    marked_white = bool(g.turn_actions.get(active, {}).get('whiteSumMarked', False))
                    if not marked_white:
                        try: g.take_penalty(active)
                        except GameError: pass

            try: g.end_turn(active)
            except GameError: break

        return score_total(g, myname)

    def _evaluate_plan(self, g: QwixxGame,
                       plan_w: Tuple[str, Optional[str]],
                       plan_c: Tuple[str, Optional[str], Optional[int]]) -> float:
        base = score_total(g, self.name)
        sim = clone(g)
        active = sim.current_player
        try: self._apply_white(sim, active, plan_w)
        except GameError: return -1e9

        if sim.phase == sim.phase.WHITE_SUM:
            for p in sim.players:
                if not sim.white_phase_complete.get(p.name, False):
                    try: sim.skip_white_sum(p.name)
                    except GameError: pass
            sim.proceed_to_color_phase_if_ready()

        try: self._apply_color(sim, active, plan_c)
        except GameError: return -1e9

        try: sim.end_turn(active)
        except GameError: pass

        sims_per = max(1, self.simulations_per_candidate)
        total = 0.0
        for _ in range(sims_per):
            sim_i = clone(sim)
            total += self._simulate_future(sim_i, self.name, self.depth_turns)
        return (total / sims_per) - base

    def plan_turn(self, g: QwixxGame):
        ws = g.white_sum()
        # Enumerate candidates (white x color), using global filters
        white_choices: List[Tuple[str, Optional[str]]] = [('skip', None)]
        white_choices += [('mark', c) for c in legal_white_sum_colors(g, self.name)]
        candidates: List[Tuple[Tuple[str,Optional[str]], Tuple[str,Optional[str],Optional[int]]]] = []

        for w_action, w_color in white_choices:
            simW = clone(g)
            white_marked = False
            try:
                if w_action == 'mark' and w_color and ws is not None:
                    simW.mark_number_white_sum(self.name, w_color, ws); simW.finalize_pending_row_closures(); white_marked = True
                else:
                    simW.skip_white_sum(self.name)
            except GameError:
                try: simW.skip_white_sum(self.name)
                except GameError: pass
            simW.phase = simW.phase.COLOR_DICE

            acts = legal_color_actions(simW, self.name)
            color_choices: List[Tuple[str, Optional[str], Optional[int]]] = [('skip', None, None)]
            color_choices += [('mark', c, wd) for (c, wd, _) in acts for _ in [(0)]]

            for c_action in color_choices:
                if (not white_marked) and (c_action[0] == 'skip'): continue
                candidates.append(((w_action, w_color), c_action))

        if not candidates:
            self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None), 'value': 0.0}
            return

        # Heuristic priors + beam
        scored: List[Tuple[float, Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str], Optional[int]]]]] = []
        for w_act, c_act in candidates:
            prior = _plan_heuristic_prior(g, self.name, w_act, c_act)
            scored.append((prior, (w_act, c_act)))
        scored.sort(key=lambda t: t[0], reverse=True)
        beam = scored[:max(1, self.beam_width)]

        # distribute simulations across beam
        beam_len = max(1, len(beam))
        self.simulations_per_candidate = max(1, self.simulations // beam_len)

        best_pair = None
        best_value = -1e9
        for _, (w_act, c_act) in beam:
            val = self._evaluate_plan(g, w_act, c_act)
            if val > best_value:
                best_value = val
                best_pair = (w_act, c_act)

        if best_pair is None:
            # fallback to the best prior
            _, (w_act, c_act) = beam[0]
            self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': -1e9}
            return
        self.pending_plan = {'white_action': best_pair[0], 'color_action': best_pair[1], 'value': best_value}

    # Choices for non-active contexts
    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        return HeuristicBot(self.name).choose_white_sum_action(g)

    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan: return self.pending_plan.get('color_action', ('skip', None, None))
        return HeuristicBot(self.name).choose_color_action(g)

# ==================== ScoutBot (hybrid: heuristic first, tiny MCTS for close calls) ====================
@dataclass
class ScoutBot:
    name: str
    top_k: int = 5
    sims_per: int = 40
    depth_turns: int = 2
    rollout_kind: str = 'heur'
    margin_thresh: float = 0.75
    pending_plan: Optional[Dict] = field(default=None, repr=False)

    def plan_turn(self, g: QwixxGame):
        ws = g.white_sum()
        white_choices: List[Tuple[str, Optional[str]]] = [('skip', None)]
        white_choices += [('mark', c) for c in legal_white_sum_colors(g, self.name)]
        candidates: List[Tuple[Tuple[str,Optional[str]], Tuple[str,Optional[str],Optional[int]]]] = []

        # enumerate safely filtered candidates
        for w_action, w_color in white_choices:
            simW = clone(g)
            white_marked = False
            try:
                if w_action == 'mark' and w_color and ws is not None:
                    simW.mark_number_white_sum(self.name, w_color, ws); simW.finalize_pending_row_closures(); white_marked = True
                else:
                    simW.skip_white_sum(self.name)
            except GameError:
                try: simW.skip_white_sum(self.name)
                except GameError: pass

            simW.phase = simW.phase.COLOR_DICE
            acts = legal_color_actions(simW, self.name)
            color_choices: List[Tuple[str, Optional[str], Optional[int]]] = [('skip', None, None)]
            color_choices += [('mark', c, wd) for (c, wd, _) in acts for _ in [(0)]]
            for c_action in color_choices:
                if (not white_marked) and (c_action[0] == 'skip'): continue
                candidates.append(((w_action, w_color), c_action))

        if not candidates:
            self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None), 'value': 0.0}
            return

        # Rank by prior
        scored: List[Tuple[float, Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str], Optional[int]]]]] = []
        for w_act, c_act in candidates:
            p = _plan_heuristic_prior(g, self.name, w_act, c_act)
            scored.append((p, (w_act, c_act)))
        scored.sort(key=lambda t: t[0], reverse=True)

        if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= self.margin_thresh:
            w_act, c_act = scored[0][1]
            self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': scored[0][0]}
            return

        # tiny MCTS on top-K
        topK = scored[:max(1, self.top_k)]
        mini = MCTSBot(
            name=self.name,
            simulations=self.sims_per * max(1, len(topK)),
            depth_turns=self.depth_turns,
            beam_width=len(topK),
            rollout_kind=self.rollout_kind
        )
        mini.simulations_per_candidate = self.sims_per

        best_pair = None
        best_val = -1e9
        for _, (w_act, c_act) in topK:
            v = mini._evaluate_plan(g, w_act, c_act)
            if v > best_val:
                best_val = v
                best_pair = (w_act, c_act)

        if best_pair is None:
            w_act, c_act = scored[0][1]
            self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': scored[0][0]}
            return

        self.pending_plan = {'white_action': best_pair[0], 'color_action': best_pair[1], 'value': best_val}

    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        if self.pending_plan and self.pending_plan.get('white_action'): return self.pending_plan['white_action']
        return HeuristicBot(self.name).choose_white_sum_action(g)

    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan and self.pending_plan.get('color_action'): return self.pending_plan['color_action']
        return HeuristicBot(self.name).choose_color_action(g)

# ==================== LearnerBot (uses global filters & heuristic prior) ====================
@dataclass
class LearnerBot:
    name: str
    epsilon: float = 0.10
    store_path: str = './qwixx_data/learner.json'
    prior_blend: float = 0.30
    pending_plan: Optional[Dict] = field(default=None, repr=False)

    _store: ExperienceStore = field(default_factory=ExperienceStore, init=False, repr=False)
    _hasher: FeatureHasher = field(default_factory=FeatureHasher, init=False, repr=False)
    _decisions: List[Tuple[str, str]] = field(default_factory=list, init=False, repr=False)
    _last_choice: Optional[Tuple[str, str]] = field(default=None, init=False, repr=False)
    _last_score_before: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._store.path = self.store_path
        self._store.load()

    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        if self.pending_plan and self.pending_plan.get('white_action'):
            return self.pending_plan['white_action']
        return HeuristicBot(self.name).choose_white_sum_action(g)

    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan and self.pending_plan.get('color_action'):
            return self.pending_plan['color_action']
        return HeuristicBot(self.name).choose_color_action(g)

    def _evaluate_pair_with_prior(self, g: QwixxGame, feature_key: str,
                                  w_act: Tuple[str, Optional[str]],
                                  c_act: Tuple[str, Optional[str], Optional[int]]) -> float:
        akey = plan_key(w_act, c_act)
        q_est = self._store.expected(feature_key, akey, default=0.0)

        base = score_total(g, self.name)
        sim = clone(g)
        val = 0.0
        try:
            if w_act[0] == 'mark' and w_act[1]:
                ws = sim.white_sum()
                sim.mark_number_white_sum(self.name, w_act[1], ws)
                sim.finalize_pending_row_closures()
            else:
                sim.skip_white_sum(self.name)

            if sim.phase == sim.phase.WHITE_SUM:
                for p in sim.players:
                    if not sim.white_phase_complete.get(p.name, False):
                        try: sim.skip_white_sum(p.name)
                        except GameError: pass
                sim.proceed_to_color_phase_if_ready()

            if c_act[0] == 'mark' and c_act[1] and c_act[2]:
                d = sim.dice
                number = (d['white1'] + d[c_act[1]]) if c_act[2] == 1 else (d['white2'] + d[c_act[1]])
                sim.select_dice_for_color_phase(self.name, c_act[2], c_act[1])
                sim.mark_number_color_dice(self.name, c_act[1], number)

            val = score_total(sim, self.name) - base
        except GameError:
            val = -1e9

        return q_est + self.prior_blend * (val / 15.0)

    def plan_turn(self, g: QwixxGame):
        ws = g.white_sum()
        feature_key = self._hasher.hash_state(g, self.name)

        white_choices: List[Tuple[str, Optional[str]]] = [('skip', None)]
        white_choices += [('mark', c) for c in legal_white_sum_colors(g, self.name)]
        candidates: List[Tuple[Tuple[str,Optional[str]], Tuple[str,Optional[str],Optional[int]]]] = []

        for w_action, w_color in white_choices:
            simW = clone(g)
            white_marked = False
            try:
                if w_action == 'mark' and w_color and ws is not None:
                    simW.mark_number_white_sum(self.name, w_color, ws)
                    simW.finalize_pending_row_closures(); white_marked = True
                else:
                    simW.skip_white_sum(self.name)
            except GameError:
                try: simW.skip_white_sum(self.name)
                except GameError: pass

            simW.phase = simW.phase.COLOR_DICE
            acts = legal_color_actions(simW, self.name)
            color_choices: List[Tuple[str, Optional[str], Optional[int]]] = [('skip', None, None)]
            color_choices += [('mark', c, wd) for (c, wd, _) in acts for _ in [(0)]]

            for c_action in color_choices:
                if (not white_marked) and (c_action[0] == 'skip'):
                    continue
                candidates.append(((w_action, w_color), c_action))

        if not candidates:
            self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None), 'value': 0.0}
            return

        if random.random() < self.epsilon:
            choice = random.choice(candidates)
            score = 0.0
        else:
            best = None; best_val = -1e9
            for w_act, c_act in candidates:
                val = self._evaluate_pair_with_prior(g, feature_key, w_act, c_act)
                if val > best_val:
                    best, best_val = (w_act, c_act), val
            choice = best
            score = best_val

        w_act, c_act = choice
        self._decisions.append((feature_key, plan_key(w_act, c_act)))
        self._last_choice = (feature_key, plan_key(w_act, c_act))
        self._last_score_before = score_total(g, self.name)
        self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': score}

    def notify_step_outcome(self, g_after: QwixxGame):
        if not self._last_choice:
            return
        fk, ak = self._last_choice
        before = self._last_score_before if self._last_score_before is not None else 0
        after = score_total(g_after, self.name)
        delta = after - before
        reward = delta / 15.0
        self._store.update(fk, ak, reward)
        self._last_choice = None
        self._last_score_before = None

    def notify_game_end(self, final_scores: Dict[str, int], save: bool = True):
        reward = default_reward_from_scores(final_scores, self.name)
        for fk, ak in self._decisions:
            self._store.update(fk, ak, reward)
        self._decisions.clear()
        if save:
            self._store.save()
