# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
import copy, random

from .game import QwixxGame, Player, COLORS, GameError
from .render import sequence_valid, ROW_NUMBERS
from .learn import ExperienceStore, FeatureHasher, plan_key, default_reward_from_scores

__all__ = [
    # Core heuristics
    'HeuristicBot', 'HeuristicStrongBot',
    # Profiles
    'HeurProfileBot', 'HEUR_PROFILES', 'register_heur_profile', 'bulk_register_random_profiles',
    # Gap family
    'GapPolicyBot', 'Gap1Bot', 'Gap1LockBot', 'Gap2Bot', 'Gap2LockBot', 'Gap3Bot', 'Gap3LockBot',
    # Probability-aware gaps (old/new)
    'ProbGapBot', 'ProbGapLockBot', 'ProbGap2Bot', 'ProbGap2LockBot',
    # adeptive gapbots
    'GapAdaptive1Bot', 'GapAdaptive1LockBot',
    'GapAdaptive2Bot', 'GapAdaptive2LockBot',
    'GapAdaptive3Bot', 'GapAdaptive3LockBot',
    # Search / hybrid
    'MCTSBot', 'ScoutBot',
    # Learner
    'LearnerBot',
    # Shared helpers for tournament code
    'legal_white_sum_colors', 'legal_color_actions',
    # tunables
    'MIN_COMPONENT_VALUE', 'COMBO_DELTA_MARGIN'
    
    
]

# ==================== Heuristic defaults ====================
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

# ==================== Basic helpers ====================
def score_total(g: QwixxGame, name: str) -> int:
    return g.calculate_score(name)['total']

def clone(g: QwixxGame) -> QwixxGame:
    return copy.deepcopy(g)

def last_number(color: str) -> int:
    return 12 if color in ('red', 'yellow') else 2

def frontier(row: List[int], color: str) -> Optional[int]:
    if not row: return None
    return max(row) if color in ('red', 'yellow') else min(row)

def sorted_insert_preview(row: List[int], color: str, number: int) -> List[int]:
    if number in row: return list(row)
    r2 = list(row) + [number]
    if color in ('red', 'yellow'): r2.sort()
    else: r2.sort(reverse=True)
    return r2

def big_skip_cells(row: List[int], color: str, new_num: int) -> int:
    f = frontier(row, color)
    if f is None: return 0
    if color in ('red', 'yellow'): return max(0, new_num - (f + 1))
    else: return max(0, (f - 1) - new_num)

def skip_from_start(row: List[int], color: str, new_num: int) -> int:
    if row: return big_skip_cells(row, color, new_num)
    return max(0, new_num - 2) if color in ('red', 'yellow') else max(0, 12 - new_num)

def max_possible_marks_after(row_after: List[int], color: str) -> int:
    n = len(row_after)
    if not row_after: return 11
    if color in ('red', 'yellow'):
        hi = max(row_after)
        future = max(0, 12 - hi)
    else:
        lo = min(row_after)
        future = max(0, lo - 2)
    return n + future

def infeasible_to_lock(row_after: List[int], color: str) -> bool:
    return max_possible_marks_after(row_after, color) < 6

def lock_feasible_after(row_before: List[int], color: str, number: int) -> bool:
    return not infeasible_to_lock(sorted_insert_preview(row_before, color, number), color)

def is_last_pick(color: str, number: int) -> bool:
    return number == last_number(color)

def last_pick_allowed(row_before: List[int], color: str, number: int) -> bool:
    return (not is_last_pick(color, number)) or (len(row_before) >= 5)

def flexibility_score(row_after: List[int], color: str) -> int:
    if not row_after: return 11
    return max(0, (12 - max(row_after)) if color in ('red', 'yellow') else (min(row_after) - 2))

def will_close_now(g_after: QwixxGame, color: str, actor: str) -> bool:
    return g_after.closed_rows[color] and (actor in g_after.row_closers[color])

# --- penalties & misc ---
def penalties_of(g: QwixxGame, name: str) -> int:
    p = next(p for p in g.players if p.name == name)
    return getattr(p, 'penalties', 0)

def max_other_score(g: QwixxGame, me: str) -> int:
    mx = -10**9
    for p in g.players:
        if p.name == me: continue
        s = score_total(g, p.name)
        if s > mx: mx = s
    return 0 if mx == -10**9 else mx

def marks_in_row(row: List[int]) -> int:
    return len(row)

# ==================== Global legal action filters ====================
def legal_white_sum_colors(g: QwixxGame, name: str) -> List[str]:
    ws = g.white_sum()
    if ws is None or g.phase != g.phase.WHITE_SUM: return []
    pl = next(p for p in g.players if p.name == name)
    out: List[str] = []
    for color in COLORS:
        if g.closed_rows[color]: continue
        row = getattr(pl, color)
        if ws not in ROW_NUMBERS[color] or ws in row: continue
        if not sequence_valid(row, color, ws): continue
        if not last_pick_allowed(row, color, ws): continue
        if not lock_feasible_after(row, color, ws): continue
        out.append(color)
    return out

def legal_color_actions(g: QwixxGame, name: str) -> List[Tuple[str,int,int]]:
    if g.phase != g.phase.COLOR_DICE: return []
    pl = next(p for p in g.players if p.name == name)
    d = g.dice
    if d['white1'] is None or d['white2'] is None: return []
    actions: List[Tuple[str,int,int]] = []
    for color in COLORS:
        if g.closed_rows[color] or not g.active_dice[color]: continue
        row = getattr(pl, color)
        n1 = d['white1'] + d[color]
        if (n1 not in row) and sequence_valid(row, color, n1) and last_pick_allowed(row, color, n1) and lock_feasible_after(row, color, n1):
            actions.append((color, 1, n1))
        n2 = d['white2'] + d[color]
        if (n2 not in row) and sequence_valid(row, color, n2) and last_pick_allowed(row, color, n2) and lock_feasible_after(row, color, n2):
            actions.append((color, 2, n2))
    return actions

# ====== dice mass (2d6) ======
_2D6_COUNTS = {2:1,3:2,4:3,5:4,6:5,7:6,8:5,9:4,10:3,11:2,12:1}
_PEAK = _2D6_COUNTS[7]
_D6NORM = {n: _2D6_COUNTS[n]/_PEAK for n in range(2,13)}  # normalize by peak (7)

def _skipped_integers(row: List[int], color: str, new_num: int) -> Iterable[int]:
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

CENTER_BAND = (6, 8)

def crosses_center(row: List[int], color: str, new_num: int) -> bool:
    for n in _skipped_integers(row, color, new_num):
        if CENTER_BAND[0] <= n <= CENTER_BAND[1]:
            return True
    return False

# ==================== Base HeuristicBot ====================
@dataclass
class HeuristicBot:
    name: str
    close_now_bonus: float = CLOSE_NOW_BONUS
    setup_bonus: float = SETUP_BONUS
    flex_weight: float = FLEX_WEIGHT
    big_skip_penalty_per_cell: float = BIG_SKIP_PENALTY_PER_CELL
    no_lock_cap_penalty: float = NO_LOCK_CAP_PENALTY
    min_component_value: float = MIN_COMPONENT_VALUE
    combo_delta_margin: float = COMBO_DELTA_MARGIN

    pending_plan: Optional[Dict] = field(default=None, repr=False)

    def _score_component(
        self, base: int, sim: QwixxGame, row_before: List[int],
        row_after: List[int], color: str, placed_number: int, delta: float,
        marked_white: bool = True
    ) -> float:
        skipped = big_skip_cells(row_before, color, placed_number)
        val = float(delta)
        if not marked_white: val += 0.5
        val -= self.big_skip_penalty_per_cell * skipped
        if will_close_now(sim, color, self.name): val += self.close_now_bonus
        if len(row_after) >= 4 and (last_number(color) not in row_after) and skipped == 0:
            val += self.setup_bonus
        if infeasible_to_lock(row_after, color): val -= self.no_lock_cap_penalty
        val += self.flex_weight * flexibility_score(row_after, color)
        return val

    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        if self.pending_plan and self.pending_plan.get('white_action'):
            return self.pending_plan['white_action']
        options = legal_white_sum_colors(g, self.name)
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
                val = self._score_component(base, sim, row_before, row_after, color, ws, delta, marked_white=True)
                if val > best_val: best_val, best = val, ('mark', color)
            except GameError:
                continue
        return best

    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan and self.pending_plan.get('color_action'):
            return self.pending_plan['color_action']
        marked_white = bool(g.turn_actions.get(self.name, {}).get('whiteSumMarked', False))
        actions = legal_color_actions(g, self.name)
        if not actions: return ('skip', None, None)
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
                val = self._score_component(base, sim, row_before, row_after, color, number, delta, marked_white=marked_white)
                if val > best_val: best_val, best = val, ('mark', color, wd)
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
            return self._score_component(base, sim_afterW, row_before, row_after, color_w, ws, delta_white, marked_white=True)

        def c_value(row_before, color_c, wd, number, sim_afterC, delta_color) -> float:
            row_after = getattr(next(p for p in sim_afterC.players if p.name == self.name), color_c)
            marked_white_local = bool(sim_afterC.turn_actions.get(self.name, {}).get('whiteSumMarked', False))
            return self._score_component(base, sim_afterC, row_before, row_after, color_c, number, delta_color, marked_white=marked_white_local)

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
                    white_marked = False; delta_white = 0.0; w_component_val = 0.0
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
                if (not white_marked) and (act_c == 'skip'): continue
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

        # Penalty endgame tactic: up by >=6 and already 3 penalties -> take 4th to end game
        my_pen = penalties_of(g, self.name)
        lead = score_total(g, self.name) - max_other_score(g, self.name)
        if my_pen >= 3 and lead >= 6:
            self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None), 'value': 0.0}
            return

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
            if w_comp < self.min_component_value and (total - best_single_val) < self.combo_delta_margin:
                plan_w = ('skip', None); total = c_comp
            elif c_comp < self.min_component_value and (total - best_single_val) < self.combo_delta_margin:
                plan_c = ('skip', None, None); total = w_comp
        self.pending_plan = {'white_action': plan_w, 'color_action': plan_c, 'value': total}

# ==================== HeuristicStrongBot (phase + threat + A/B/C/E) ====================
PHASE_DELTA = {
    'early': {'flex_weight': +0.05, 'big_skip_penalty_per_cell': -0.15, 'setup_bonus': +0.25, 'close_now_bonus': +0.0, 'no_lock_cap_penalty': +0.0},
    'mid':   {'flex_weight': +0.02, 'big_skip_penalty_per_cell': -0.10, 'setup_bonus': +0.75, 'close_now_bonus': +0.0, 'no_lock_cap_penalty': +0.0},
    'lock':  {'flex_weight': +0.00, 'big_skip_penalty_per_cell': +0.00, 'setup_bonus': +0.25, 'close_now_bonus': +1.00, 'no_lock_cap_penalty': +1.00},
}

THREAT_WEIGHTS = {
    'deny_bonus': +0.8,
    'exposure_penalty': -0.6,
    'center_exposure': -0.3,
    'max_total': 1.6,
}

def _lock_approach(row_before: List[int], row_after: List[int], color: str) -> bool:
    return (len(row_before) >= 5) or (max_possible_marks_after(row_after, color) <= 7)

def _phase_bucket(row_before: List[int], row_after: List[int], color: str) -> str:
    if _lock_approach(row_before, row_after, color): return 'lock'
    n = len(row_before)
    if n <= 2: return 'early'
    if 3 <= n <= 4: return 'mid'
    return 'lock'

def _distance_to_last(color: str, frontier_num: Optional[int]) -> Optional[int]:
    if frontier_num is None: return None
    last = last_number(color)
    if color in ('red','yellow'):
        return max(0, last - (frontier_num + 1))
    else:
        return max(0, (frontier_num - 1) - last)

class HeuristicStrongBot(HeuristicBot):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            close_now_bonus=CLOSE_NOW_BONUS,
            setup_bonus=4.5,
            flex_weight=0.18,
            big_skip_penalty_per_cell=1.10,
            no_lock_cap_penalty=NO_LOCK_CAP_PENALTY,
            min_component_value=MIN_COMPONENT_VALUE,
            combo_delta_margin=COMBO_DELTA_MARGIN
        )

    def _local_weights(
        self,
        sim_after: QwixxGame,
        row_before: List[int],
        row_after: List[int],
        color: str,
        placed_number: int,
        crossed_center: bool
    ) -> Dict[str, float]:
        bucket = _phase_bucket(row_before, row_after, color)
        delta = PHASE_DELTA[bucket].copy()
        if bucket == 'mid' and crossed_center:
            delta['big_skip_penalty_per_cell'] = 0.0
        return {
            'close_now_bonus': self.close_now_bonus + delta['close_now_bonus'],
            'setup_bonus': self.setup_bonus + delta['setup_bonus'],
            'flex_weight': self.flex_weight + delta['flex_weight'],
            'big_skip_penalty_per_cell': max(0.0, self.big_skip_penalty_per_cell + delta['big_skip_penalty_per_cell']),
            'no_lock_cap_penalty': self.no_lock_cap_penalty + delta['no_lock_cap_penalty'],
        }

    def _threat_term(
        self,
        sim_after: QwixxGame,
        color: str,
        actor: str,
        crossed_center: bool
    ) -> float:
        if sim_after.closed_rows[color]: return 0.0
        deny = 0.0; expose = 0.0
        i_closed = will_close_now(sim_after, color, actor)
        for p in sim_after.players:
            if p.name == actor: continue
            row_op = getattr(p, color)
            if len(row_op) < 4: continue
            f = frontier(row_op, color)
            dist = _distance_to_last(color, f)
            if dist is None: continue
            # E) scale deny/exposure if opponent already has ≥5 marks
            scale = 1.3 if len(row_op) >= 5 else 1.0
            if dist <= 1:
                if i_closed: deny += scale * THREAT_WEIGHTS['deny_bonus']
                else:        expose += scale * (-THREAT_WEIGHTS['exposure_penalty'])
        if (not i_closed) and crossed_center:
            expose += (-THREAT_WEIGHTS['center_exposure'])
        net = (deny - expose)
        if net > 0: net = min(net, THREAT_WEIGHTS['max_total'])
        else:       net = max(net, -THREAT_WEIGHTS['max_total'])
        return net

    def _score_component(
        self, base: int, sim: QwixxGame, row_before: List[int],
        row_after: List[int], color: str, placed_number: int, delta: float,
        marked_white: bool = True
    ) -> float:
        crossed_center = crosses_center(row_before, color, placed_number)
        w = self._local_weights(sim, row_before, row_after, color, placed_number, crossed_center)

        skipped = big_skip_cells(row_before, color, placed_number)
        val = float(delta)
        if not marked_white: val += 0.5
        val -= w['big_skip_penalty_per_cell'] * skipped
        if will_close_now(sim, color, self.name): val += w['close_now_bonus']
        if len(row_after) >= 4 and (last_number(color) not in row_after) and skipped == 0:
            val += w['setup_bonus']
        if infeasible_to_lock(row_after, color): val -= w['no_lock_cap_penalty']
        val += w['flex_weight'] * flexibility_score(row_after, color)

        # --- A) soft gap cap ---
        skipped_ints = list(_skipped_integers(row_before, color, placed_number))
        pw = sum(_D6NORM.get(n, 1.0) for n in skipped_ints)
        if len(row_before) <= 2: cap = 1.6
        elif len(row_before) <= 4: cap = 1.8
        else: cap = 1.2
        if pw > cap:
            val -= 0.8 * (pw - cap)  # tune

        # --- B) center band retention unless closing ---
        center_skipped = sum(1 for n in skipped_ints if 6 <= n <= 8)
        if center_skipped and not will_close_now(sim, color, self.name):
            val -= 0.5 * center_skipped  # tune

        # --- C) headroom preference ---
        mp = max_possible_marks_after(row_after, color)
        if len(row_before) <= 2 and mp < 8: val -= 0.5
        elif 3 <= len(row_before) <= 4 and mp < 7: val -= 0.3

        # Threat awareness (+E scaling)
        val += self._threat_term(sim, color, self.name, crossed_center)
        return val

# ==================== HEUR profiles ====================
HEUR_PROFILES: Dict[str, Dict[str, float]] = {}

def register_heur_profile(name: str, params: Dict[str, float]) -> None:
    HEUR_PROFILES[name] = dict(params)

class HeurProfileBot(HeuristicBot):
    def __init__(self, name: str, profile: str):
        p = HEUR_PROFILES.get(profile, {})
        super().__init__(
            name=name,
            close_now_bonus=p.get('close_now_bonus', CLOSE_NOW_BONUS),
            setup_bonus=p.get('setup_bonus', SETUP_BONUS),
            flex_weight=p.get('flex_weight', FLEX_WEIGHT),
            big_skip_penalty_per_cell=p.get('big_skip_penalty_per_cell', BIG_SKIP_PENALTY_PER_CELL),
            no_lock_cap_penalty=p.get('no_lock_cap_penalty', NO_LOCK_CAP_PENALTY),
            min_component_value=p.get('min_component_value', MIN_COMPONENT_VALUE),
            combo_delta_margin=p.get('combo_delta_margin', COMBO_DELTA_MARGIN),
        )

def bulk_register_random_profiles(
    count: int = 8,
    seed: Optional[int] = None,
    ranges: Optional[Dict[str, Tuple[float,float]]] = None,
    prefix: str = "rand"
) -> List[str]:
    if ranges is None:
        ranges = {
            'setup_bonus': (3.5, 6.0),
            'flex_weight': (0.12, 0.26),
            'big_skip_penalty_per_cell': (0.90, 1.30),
            'close_now_bonus': (8.0, 11.0),
            'no_lock_cap_penalty': (5.0, 8.0),
            'min_component_value': (MIN_COMPONENT_VALUE, MIN_COMPONENT_VALUE),
            'combo_delta_margin': (COMBO_DELTA_MARGIN, COMBO_DELTA_MARGIN),
        }
    rng = random.Random(seed) if seed is not None else random
    def pick(lo, hi): return round(rng.uniform(lo, hi), 2)
    names: List[str] = []
    for i in range(count):
        params = {k: pick(*ranges[k]) for k in ranges}
        name = f"{prefix}{i+1:02d}"
        register_heur_profile(name, params)
        names.append(name)
    return names

# ==================== Gap policy family ====================
@dataclass
class GapPolicyBot(HeuristicBot):
    max_skip: int = 1
    allow_break_for_lock: bool = False

    def _would_lock(self, row_before: List[int], color: str, number: int) -> bool:
        return is_last_pick(color, number) and (len(row_before) + 1 >= 6)

    def _allowed_by_skip(self, row_before: List[int], color: str, number: int) -> bool:
        raw = skip_from_start(row_before, color, number)
        allowance = self.max_skip
        invested = len(row_before) >= 4
        x_center = crosses_center(row_before, color, number)
        will_lock = self._would_lock(row_before, color, number)
        after = sorted_insert_preview(row_before, color, number)
        near_lock_tight = (not infeasible_to_lock(after, color)) and (max_possible_marks_after(after, color) <= 6)

        if invested and not x_center: allowance += 1
        if x_center: allowance -= 1
        if near_lock_tight and not will_lock: allowance -= 1

        allowance = max(0, allowance)
        if raw <= allowance: return True
        return self.allow_break_for_lock and will_lock

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
                val = HeuristicBot._score_component(self, base, sim, row_before, row_after, color, ws, delta, marked_white=True)
                if val > best_val: best_val, best = val, ('mark', color)
            except GameError: continue
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
                val = HeuristicBot._score_component(self, base, sim, row_before, row_after, color, num, delta, marked_white=marked_white)
                if val > best_val: best_val, best = val, ('mark', color, wd)
            except GameError: continue
        return best

    # PLANNING override: enforce GAP policy in planning and apply penalties leniency
    def plan_turn(self, g: QwixxGame):
        base = score_total(g, self.name)
        ws = g.white_sum()

        # If I'm losing and already at 3 penalties, be more lenient to avoid the 4th
        my_pen = penalties_of(g, self.name)
        losing = (score_total(g, self.name) < max_other_score(g, self.name))
        extra_allowance = 0
        break_any = False
        if my_pen >= 3 and losing:
            extra_allowance = 2
            break_any = True

        # temporarily bump allowances
        old_max = self.max_skip
        old_break = self.allow_break_for_lock
        self.max_skip = self.max_skip + extra_allowance
        self.allow_break_for_lock = self.allow_break_for_lock or break_any

        white_choices: List[Tuple[str, Optional[str]]] = [('skip', None)]
        white_choices += [('mark', c) for c in self._filtered_white_options(g)]

        best_pair = None
        best_total_val = -1e9
        best_white_only_val = -1e9
        best_color_only_val = -1e9
        best_white_only: Tuple[str, Optional[str]] = ('skip', None)
        best_color_only: Tuple[str, Optional[str], Optional[int]] = ('skip', None, None)

        def w_value(row_before, color_w, sim_afterW, delta_white) -> float:
            row_after = getattr(next(p for p in sim_afterW.players if p.name == self.name), color_w)
            return HeuristicBot._score_component(self, base, sim_afterW, row_before, row_after, color_w, ws, delta_white, marked_white=True)

        def c_value(row_before, color_c, wd, number, sim_afterC, delta_color) -> float:
            row_after = getattr(next(p for p in sim_afterC.players if p.name == self.name), color_c)
            marked_white_local = bool(sim_afterC.turn_actions.get(self.name, {}).get('whiteSumMarked', False))
            return HeuristicBot._score_component(self, base, sim_afterC, row_before, row_after, color_c, number, delta_color, marked_white=marked_white_local)

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
                    white_marked = False; delta_white = 0.0; w_component_val = 0.0
            else:
                try: simW.skip_white_sum(self.name)
                except GameError: pass

            simW.phase = simW.phase.COLOR_DICE
            acts = self._filtered_color_actions(simW)
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
                if (not white_marked) and (act_c == 'skip'): continue
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
                    }

        # restore policy caps
        self.max_skip = old_max
        self.allow_break_for_lock = old_break

        if best_pair is None:
            if best_color_only_val >= best_white_only_val and best_color_only_val > -1e9:
                self.pending_plan = {'white_action': ('skip', None), 'color_action': best_color_only}
            elif best_white_only_val > -1e9:
                self.pending_plan = {'white_action': best_white_only, 'color_action': ('skip', None, None)}
            else:
                self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None)}
            return

        # If I'm up by ≥6 and at 3 penalties -> cash out to end
        if my_pen >= 3 and (score_total(g, self.name) - max_other_score(g, self.name)) >= 6:
            self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None)}
            return

        self.pending_plan = best_pair

# concrete gaps
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

# ==================== ProbGap (old/new) ====================
@dataclass
class GapPolicyBotBase(HeuristicBot):
    max_skip: int = 1
    allow_break_for_lock: bool = False
    def _would_lock(self, row_before: List[int], color: str, number: int) -> bool:
        return is_last_pick(color, number) and (len(row_before) + 1 >= 6)
    def _allowed_by_skip(self, row_before: List[int], color: str, number: int) -> bool:
        raw = skip_from_start(row_before, color, number)
        if raw <= self.max_skip: return True
        return self.allow_break_for_lock and self._would_lock(row_before, color, number)
    def _filtered_white_options(self, g: QwixxGame) -> List[str]:
        ws = g.white_sum()
        opts = legal_white_sum_colors(g, self.name)
        return [c for c in opts if self._allowed_by_skip(getattr(next(p for p in g.players if p.name == self.name), c), c, ws)]
    def _filtered_color_actions(self, g: QwixxGame) -> List[Tuple[str,int,int]]:
        acts = legal_color_actions(g, self.name)
        out: List[Tuple[str,int,int]] = []
        for color, wd, num in acts:
            row = getattr(next(p for p in g.players if p.name == self.name), color)
            if self._allowed_by_skip(row, color, num): out.append((color, wd, num))
        return out

@dataclass
class ProbGapBot(GapPolicyBotBase):
    max_eff_skip: float = 1.75
    def _effective_skip_weight(self, row_before: List[int], color: str, number: int) -> float:
        skipped = list(_skipped_integers(row_before, color, number))
        return sum(_D6NORM.get(n, 1.0) for n in skipped)
    def _allowed_by_skip(self, row_before: List[int], color: str, number: int) -> bool:
        if not super()._allowed_by_skip(row_before, color, number): return False
        w = self._effective_skip_weight(row_before, color, number)
        if w <= self.max_eff_skip: return True
        return self.allow_break_for_lock and self._would_lock(row_before, color, number)

class ProbGapLockBot(ProbGapBot):
    def __init__(self, name: str, max_eff_skip: float = 1.75):
        super().__init__(name=name, max_skip=2, allow_break_for_lock=True, max_eff_skip=max_eff_skip)

@dataclass
class ProbGap2Bot(GapPolicyBot):
    max_eff_skip: float = 2.0
    center_penalty: float = 1.35
    def _effective_skip_weight(self, row_before: List[int], color: str, number: int) -> float:
        skipped = list(_skipped_integers(row_before, color, number))
        base = sum(_D6NORM.get(n, 1.0) for n in skipped)
        if any(6 <= n <= 8 for n in skipped): base *= self.center_penalty
        return base
    def _allowed_by_skip(self, row_before: List[int], color: str, number: int) -> bool:
        if not super()._allowed_by_skip(row_before, color, number): return False
        w = self._effective_skip_weight(row_before, color, number)
        if w <= self.max_eff_skip: return True
        return self.allow_break_for_lock and self._would_lock(row_before, color, number)

class ProbGap2LockBot(ProbGap2Bot):
    def __init__(self, name: str, max_eff_skip: float = 2.0, center_penalty: float = 1.35):
        super().__init__(name=name, max_skip=2, allow_break_for_lock=True,
                         max_eff_skip=max_eff_skip, center_penalty=center_penalty)

# ==================== Adaptive GAP policy (parameterized) ====================
@dataclass
class GapAdaptiveBaseBot(GapPolicyBot):
    """
    Adaptive GAP policy:
      - Base GAP cap (1/2/3) and lock behavior (lock or non-lock) come from ctor.
      - At 2 penalties: +1 to raw gap allowance (more lenient than normal).
      - At 3 penalties: unlimited gap (do anything) to find a mark (avoid 4th).
      - NEVER take a 4th penalty if it would lose the game:
            if my_pen==3 and (score_total(self)-5) < max_other_score, then force a mark if any exists.
    """
    def __init__(self, name: str, base_max_skip: int, lock_variant: bool):
        super().__init__(name=name, max_skip=base_max_skip, allow_break_for_lock=lock_variant)

    # -- dynamic allowances helper --
    def _dynamic_allowances(self, g: QwixxGame) -> Tuple[int, bool]:
        my_pen = penalties_of(g, self.name)
        # Start with the bot's current caps
        dyn_max = self.max_skip
        dyn_break = self.allow_break_for_lock
        if my_pen >= 3:
            # At 3 penalties: do anything (avoid the 4th)
            dyn_max = 99
            dyn_break = True
        elif my_pen >= 2:
            # At 2 penalties: +1 raw gap allowance
            dyn_max = self.max_skip + 1
            # keep lock behavior as defined
        return dyn_max, dyn_break

    # -- choose_* should respect dynamic allowances even for non-active white --
    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        if self.pending_plan and self.pending_plan.get('white_action'):
            return self.pending_plan['white_action']
        dyn_max, dyn_break = self._dynamic_allowances(g)
        old_max, old_break = self.max_skip, self.allow_break_for_lock
        self.max_skip, self.allow_break_for_lock = dyn_max, dyn_break
        try:
            return GapPolicyBot.choose_white_sum_action(self, g)
        finally:
            self.max_skip, self.allow_break_for_lock = old_max, old_break

    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan and self.pending_plan.get('color_action'):
            return self.pending_plan['color_action']
        dyn_max, dyn_break = self._dynamic_allowances(g)
        old_max, old_break = self.max_skip, self.allow_break_for_lock
        self.max_skip, self.allow_break_for_lock = dyn_max, dyn_break
        try:
            return GapPolicyBot.choose_color_action(self, g)
        finally:
            self.max_skip, self.allow_break_for_lock = old_max, old_break

    # -- planning with penalties-aware allowances and 4th-penalty safeguard --
    def plan_turn(self, g: QwixxGame):
        base = score_total(g, self.name)
        opp_top = max_other_score(g, self.name)
        my_pen = penalties_of(g, self.name)
        lead_after_4th = (base - 5) - opp_top  # would I still lead after a 4th penalty?

        # Apply dynamic allowances for this turn
        dyn_max, dyn_break = self._dynamic_allowances(g)
        old_max, old_break = self.max_skip, self.allow_break_for_lock
        self.max_skip, self.allow_break_for_lock = dyn_max, dyn_break

        ws = g.white_sum()
        white_choices: List[Tuple[str, Optional[str]]] = [('skip', None)]
        white_choices += [('mark', c) for c in self._filtered_white_options(g)]

        best_pair = None
        best_total_val = -1e9
        best_white_only_val = -1e9
        best_color_only_val = -1e9
        best_white_only: Tuple[str, Optional[str]] = ('skip', None)
        best_color_only: Tuple[str, Optional[str], Optional[int]] = ('skip', None, None)

        def w_value(row_before, color_w, sim_afterW, delta_white) -> float:
            row_after = getattr(next(p for p in sim_afterW.players if p.name == self.name), color_w)
            return HeuristicBot._score_component(self, base, sim_afterW, row_before, row_after, color_w, ws, delta_white, marked_white=True)

        def c_value(row_before, color_c, wd, number, sim_afterC, delta_color) -> float:
            row_after = getattr(next(p for p in sim_afterC.players if p.name == self.name), color_c)
            marked_white_local = bool(sim_afterC.turn_actions.get(self.name, {}).get('whiteSumMarked', False))
            return HeuristicBot._score_component(self, base, sim_afterC, row_before, row_after, color_c, number, delta_color, marked_white=marked_white_local)

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
                    white_marked = False; delta_white = 0.0; w_component_val = 0.0
            else:
                try: simW.skip_white_sum(self.name)
                except GameError: pass

            simW.phase = simW.phase.COLOR_DICE
            acts = self._filtered_color_actions(simW)
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
                if (not white_marked) and (act_c == 'skip'): continue
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
                    }

        # === NEVER lose via 4th penalty: if my_pen==3 and I'd trail after -5, force any mark ===
        if my_pen >= 3 and lead_after_4th < 0:
            # If plan is skip/skip (penalty), but any mark exists, force it
            if best_pair is None or (best_pair['white_action'][0] == 'skip' and best_pair['color_action'][0] == 'skip'):
                if best_color_only_val > -1e9:
                    self.pending_plan = {'white_action': ('skip', None), 'color_action': best_color_only}
                    self.max_skip, self.allow_break_for_lock = old_max, old_break
                    return
                if best_white_only_val > -1e9:
                    self.pending_plan = {'white_action': best_white_only, 'color_action': ('skip', None, None)}
                    self.max_skip, self.allow_break_for_lock = old_max, old_break
                    return
                # Else truly forced penalty (no mark at all) — extremely rare.

        if best_pair is None:
            if best_color_only_val >= best_white_only_val and best_color_only_val > -1e9:
                self.pending_plan = {'white_action': ('skip', None), 'color_action': best_color_only}
            elif best_white_only_val > -1e9:
                self.pending_plan = {'white_action': best_white_only, 'color_action': ('skip', None, None)}
            else:
                self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None)}
        else:
            self.pending_plan = best_pair

        # restore original caps
        self.max_skip, self.allow_break_for_lock = old_max, old_break
        
        
        
# Concrete Adaptive-1
class GapAdaptive1Bot(GapAdaptiveBaseBot):
    def __init__(self, name: str):
        super().__init__(name=name, base_max_skip=1, lock_variant=False)

class GapAdaptive1LockBot(GapAdaptiveBaseBot):
    def __init__(self, name: str):
        super().__init__(name=name, base_max_skip=1, lock_variant=True)

# Concrete Adaptive-2
class GapAdaptive2Bot(GapAdaptiveBaseBot):
    def __init__(self, name: str):
        super().__init__(name=name, base_max_skip=2, lock_variant=False)

class GapAdaptive2LockBot(GapAdaptiveBaseBot):
    def __init__(self, name: str):
        super().__init__(name=name, base_max_skip=2, lock_variant=True)

# Concrete Adaptive-3
class GapAdaptive3Bot(GapAdaptiveBaseBot):
    def __init__(self, name: str):
        super().__init__(name=name, base_max_skip=3, lock_variant=False)

class GapAdaptive3LockBot(GapAdaptiveBaseBot):
    def __init__(self, name: str):
        super().__init__(name=name, base_max_skip=3, lock_variant=True)
        
        
        
# ==================== MCTSBot (unchanged behavior) ====================
def _plan_heuristic_prior(g: QwixxGame, me: str,
                          w_act: Tuple[str, Optional[str]],
                          c_act: Tuple[str, Optional[str], Optional[int]]) -> float:
    base = score_total(g, me)
    sim = clone(g)
    try:
        if w_act[0] == 'mark' and w_act[1]:
            ws = sim.white_sum()
            sim.mark_number_white_sum(me, w_act[1], ws)
            sim.finalize_pending_row_closures()
        else:
            sim.skip_white_sum(me)
        if sim.phase == sim.phase.WHITE_SUM:
            for p in sim.players:
                if not sim.white_phase_complete.get(p.name, False):
                    try: sim.skip_white_sum(p.name)
                    except GameError: pass
            sim.proceed_to_color_phase_if_ready()
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
    rollout_kind: str = 'heur'  # 'heur' | 'gap2' | 'gap2lock' | 'probgap'
    pending_plan: Optional[Dict] = field(default=None, repr=False)

    def _rollout_policy(self, pname: str) -> HeuristicBot:
        if self.rollout_kind == 'gap2': return Gap2Bot(pname)
        if self.rollout_kind == 'gap2lock': return Gap2LockBot(pname)
        if self.rollout_kind == 'probgap': return ProbGapBot(pname)
        return HeuristicStrongBot(pname)

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
                    if getattr(bot, 'pending_plan', None) is None and hasattr(bot, 'plan_turn'):
                        bot.plan_turn(g)
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
                if getattr(bot, 'pending_plan', None) is None and hasattr(bot, 'plan_turn'):
                    bot.plan_turn(g)
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
        sims_per = max(1, getattr(self, 'simulations_per_candidate', self.simulations))
        total = 0.0
        for _ in range(sims_per):
            sim_i = clone(sim)
            total += self._simulate_future(sim_i, self.name, self.depth_turns)
        return (total / sims_per) - base

    def plan_turn(self, g: QwixxGame):
        ws = g.white_sum()
        white_choices: List[Tuple[str, Optional[str]]] = [('skip', None)]
        white_choices += [('mark', c) for c in legal_white_sum_colors(g, self.name)]
        candidates: List[Tuple[Tuple[str,Optional[str]], Tuple[str,Optional[str],Optional[int]]]] = []
        for w_action, w_color in white_choices:
            simW = clone(g); white_marked = False
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
        scored: List[Tuple[float, Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str], Optional[int]]]]] = []
        for w_act, c_act in candidates:
            prior = _plan_heuristic_prior(g, self.name, w_act, c_act)
            scored.append((prior, (w_act, c_act)))
        scored.sort(key=lambda t: t[0], reverse=True)
        beam: List[Tuple[float, Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str], Optional[int]]]]] = []
        for s, pair in scored:
            if not beam:
                beam.append((s, pair)); continue
            if len(beam) < max(1, self.beam_width) or s >= beam[0][0] - 0.25:
                beam.append((s, pair))
            if len(beam) >= max(1, self.beam_width) * 2:
                break
        beam_len = max(1, len(beam))
        self.simulations_per_candidate = max(1, self.simulations // beam_len)
        best_pair = None; best_value = -1e9
        for _, (w_act, c_act) in beam:
            val = self._evaluate_plan(g, w_act, c_act)
            if val > best_value:
                best_value = val; best_pair = (w_act, c_act)
        if best_pair is None:
            _, (w_act, c_act) = beam[0]
            self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': -1e9}
            return
        self.pending_plan = {'white_action': best_pair[0], 'color_action': best_pair[1], 'value': best_value}

    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        return HeuristicBot(self.name).choose_white_sum_action(g)
    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan: return self.pending_plan.get('color_action', ('skip', None, None))
        return HeuristicBot(self.name).choose_color_action(g)

# ==================== ScoutBot ====================
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
        for w_action, w_color in white_choices:
            simW = clone(g); white_marked = False
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
        scored: List[Tuple[float, Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str], Optional[int]]]]] = []
        for w_act, c_act in candidates:
            p = _plan_heuristic_prior(g, self.name, w_act, c_act)
            scored.append((p, (w_act, c_act)))
        scored.sort(key=lambda t: t[0], reverse=True)
        if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= self.margin_thresh:
            w_act, c_act = scored[0][1]
            self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': scored[0][0]}
            return
        mini = MCTSBot(
            name=self.name,
            simulations=self.sims_per * max(1, len(scored[:self.top_k])),
            depth_turns=self.depth_turns,
            beam_width=len(scored[:self.top_k]),
            rollout_kind=self.rollout_kind
        )
        mini.simulations_per_candidate = self.sims_per
        best_pair = None; best_val = -1e9
        for _, (w_act, c_act) in scored[:self.top_k]:
            v = mini._evaluate_plan(g, w_act, c_act)
            if v > best_val:
                best_val = v; best_pair = (w_act, c_act)
        if best_pair is None:
            w_act, c_act = scored[0][1]
            self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': scored[0][0]}
            return
        self.pending_plan = {'white_action': best_pair[0], 'color_action': best_pair[1], 'value': best_val}

    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        if self.pending_plan and self.pending_plan.get('white_action'):
            return self.pending_plan['white_action']
        return HeuristicBot(self.name).choose_white_sum_action(g)
    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan and self.pending_plan.get('color_action'):
            return self.pending_plan['color_action']
        return HeuristicBot(self.name).choose_color_action(g)


# ==================== LearnerBot ====================
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
            simW = clone(g); white_marked = False
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
                if (not white_marked) and (c_action[0] == 'skip'): continue
                candidates.append(((w_action, w_color), c_action))
        if not candidates:
            self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None), 'value': 0.0}
            return
        if random.random() < self.epsilon:
            choice = random.choice(candidates); score = 0.0
        else:
            best = None; best_val = -1e9
            for w_act, c_act in candidates:
                val = self._evaluate_pair_with_prior(g, feature_key, w_act, c_act)
                if val > best_val: best, best_val = (w_act, c_act), val
            choice = best; score = best_val
        w_act, c_act = choice
        self._decisions.append((feature_key, plan_key(w_act, c_act)))
        self._last_choice = (feature_key, plan_key(w_act, c_act))
        self._last_score_before = score_total(g, self.name)
        self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': score}

    def notify_step_outcome(self, g_after: QwixxGame):
        if not self._last_choice: return
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
