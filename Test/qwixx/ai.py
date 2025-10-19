# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy, random

from .game import QwixxGame, Player, COLORS, GameError
from .render import sequence_valid, SHORT_TO_COLOR, COLOR_SHORT, ROW_NUMBERS
from .learn import ExperienceStore, FeatureHasher, plan_key, default_reward_from_scores

# ===== Heuristic constants (as before) =====
CLOSE_NOW_BONUS = 9.0
SETUP_BONUS     = 4.0
FLEX_WEIGHT     = 0.15
BIG_SKIP_PENALTY_PER_CELL = 1.25
NO_LOCK_CAP_PENALTY = 6.0

MIN_COMPONENT_VALUE = 0.75
COMBO_DELTA_MARGIN  = 0.25

# ===== MCTS defaults (as before) =====
MCTS_SIMULATIONS  = 400
MCTS_DEPTH_TURNS  = 2

# ---------- shared helpers ----------
def score_total(g: QwixxGame, name: str) -> int:
    return g.calculate_score(name)['total']

def clone(g: QwixxGame) -> QwixxGame:
    return copy.deepcopy(g)

def legal_white_sum_colors(g: QwixxGame, name: str) -> List[str]:
    ws = g.white_sum()
    if ws is None or g.phase != g.phase.WHITE_SUM:
        return []
    pl = next(p for p in g.players if p.name == name)
    return [
        color for color in COLORS
        if not g.closed_rows[color]
        and ws in ROW_NUMBERS[color]
        and (ws not in getattr(pl, color))
        and sequence_valid(getattr(pl, color), color, ws)
    ]

def legal_color_actions(g: QwixxGame, name: str) -> List[Tuple[str,int,int]]:
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
        w1 = d['white1'] + d[color]
        if (w1 not in row) and sequence_valid(row, color, w1):
            actions.append((color, 1, w1))
        w2 = d['white2'] + d[color]
        if (w2 not in row) and sequence_valid(row, color, w2):
            actions.append((color, 2, w2))
    return actions

def last_number(color: str) -> int:
    return 12 if color in ('red','yellow') else 2

def frontier(row: List[int], color: str) -> Optional[int]:
    if not row:
        return None
    return max(row) if color in ('red','yellow') else min(row)

def big_skip_cells(row: List[int], color: str, new_num: int) -> int:
    f = frontier(row, color)
    if f is None:
        return 0
    if color in ('red','yellow'):
        return max(0, new_num - (f + 1))
    else:
        return max(0, (f - 1) - new_num)

def max_possible_marks_after(row_after: List[int], color: str) -> int:
    n_now = len(row_after)
    if not row_after:
        return 11
    if color in ('red','yellow'):
        hi = max(row_after)
        future = max(0, 12 - hi)
    else:
        lo = min(row_after)
        future = max(0, lo - 2)
    return n_now + future

def infeasible_to_lock(row_after: List[int], color: str) -> bool:
    return max_possible_marks_after(row_after, color) < 6
def setup_closure_signal(row_after: List[int], color: str, skipped_cells: int) -> bool:
    if len(row_after) < 4:
        return False
    target = last_number(color)
    if color in ('red','yellow'):
        dist = target - max(row_after)
    else:
        dist = min(row_after) - target
    return (dist <= 3) and (skipped_cells == 0) and (target not in row_after)

def flexibility_score(row_after: List[int], color: str) -> int:
    if not row_after:
        return 11
    if color in ('red','yellow'):
        return max(0, 12 - max(row_after))
    else:
        return max(0, min(row_after) - 2)

def will_close_now(g_after: QwixxGame, color: str, actor: str) -> bool:
    return g_after.closed_rows[color] and (actor in g_after.row_closers[color])

# ===== HeuristicBot (unchanged logic; used by Learner & MCTS rollouts) =====
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
                after = score_total(sim, self.name)
                delta = after - base
                row_after = getattr(next(p for p in sim.players if p.name == self.name), color)
                skipped = big_skip_cells(row_before, color, ws)
                val = delta
                val -= BIG_SKIP_PENALTY_PER_CELL * skipped
                if will_close_now(sim, color, self.name): val += CLOSE_NOW_BONUS
                if setup_closure_signal(row_after, color, skipped): val += SETUP_BONUS
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
                after = score_total(sim, self.name)
                delta = after - base
                row_after = getattr(next(p for p in sim.players if p.name == self.name), color)
                skipped = big_skip_cells(row_before, color, number)
                val = delta + (0.5 if not marked_white else 0.0)
                val -= BIG_SKIP_PENALTY_PER_CELL * skipped
                if will_close_now(sim, color, self.name): val += CLOSE_NOW_BONUS
                if setup_closure_signal(row_after, color, skipped): val += SETUP_BONUS
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

        def value_white_component(row_before, color_w, sim_afterW, delta_white) -> float:
            row_after = getattr(next(p for p in sim_afterW.players if p.name == self.name), color_w)
            skipped = big_skip_cells(row_before, color_w, ws)
            val = delta_white
            val -= BIG_SKIP_PENALTY_PER_CELL * skipped
            if will_close_now(sim_afterW, color_w, self.name): val += CLOSE_NOW_BONUS
            if setup_closure_signal(row_after, color_w, skipped): val += SETUP_BONUS
            if infeasible_to_lock(row_after, color_w): val -= NO_LOCK_CAP_PENALTY
            val += FLEX_WEIGHT * flexibility_score(row_after, color_w)
            return val

        def value_color_component(row_before, color_c, wd, number, sim_afterC, delta_color) -> float:
            row_after = getattr(next(p for p in sim_afterC.players if p.name == self.name), color_c)
            skipped = big_skip_cells(row_before, color_c, number)
            val = delta_color
            val -= BIG_SKIP_PENALTY_PER_CELL * skipped
            if will_close_now(sim_afterC, color_c, self.name): val += CLOSE_NOW_BONUS
            if setup_closure_signal(row_after, color_c, skipped): val += SETUP_BONUS
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
                    w_component_val = value_white_component(row_before_w, w_color, simW, delta_white)
                except GameError:
                    white_marked = False
                    delta_white = 0.0
                    w_component_val = 0.0
            else:
                try: simW.skip_white_sum(self.name)
                except GameError: pass

            simW.phase = simW.phase.COLOR_DICE
            color_actions = legal_color_actions(simW, self.name)
            color_choices: List[Tuple[str, Optional[str], Optional[int]]] = [('skip', None, None)]
            color_choices += [('mark', c, wd) for (c, wd, _) in color_actions
                              for (_, __, ___) in [(c, wd, 0)] if True]

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
                        c_val_only = value_color_component(row_before_c, color_c, wd_c, number, simC, delta_color_only)
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
                        c_component_val = value_color_component(row_before_c, color_c, wd_c, number, simPair, delta_color)
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
                        'best_white_only': (best_white_only, best_white_only_val),
                        'best_color_only': (best_color_only, best_color_only_val),
                    }

        if best_pair is None:
            # fall back to any single best found
            if best_color_only_val >= best_white_only_val and best_color_only_val > -1e9:
                self.pending_plan = {'white_action': ('skip', None), 'color_action': best_color_only}
            elif best_white_only_val > -1e9:
                self.pending_plan = {'white_action': best_white_only, 'color_action': ('skip', None, None)}
            else:
                self.pending_plan = {'white_action': ('skip', None), 'color_action': ('skip', None, None)}
            return

        w_act, w_val = best_pair['best_white_only']
        c_act, c_val = best_pair['best_color_only']
        plan_w, plan_c = best_pair['white_action'], best_pair['color_action']
        w_comp = best_pair['white_component_val']
        c_comp = best_pair['color_component_val']
        total = best_pair['total_val']

        if plan_w[0] == 'mark' and plan_c[0] == 'mark':
            best_single_val = max(w_val, c_val, -1e9)
            if w_comp < MIN_COMPONENT_VALUE and (total - max(best_single_val, c_val)) < COMBO_DELTA_MARGIN:
                plan_w = ('skip', None); total = c_comp
            elif c_comp < MIN_COMPONENT_VALUE and (total - max(best_single_val, w_val)) < COMBO_DELTA_MARGIN:
                plan_c = ('skip', None, None); total = w_comp

        self.pending_plan = {'white_action': plan_w, 'color_action': plan_c, 'value': total}

# ===== MCTSBot (unchanged from your last) =====
@dataclass
class MCTSBot:
    name: str
    simulations: int = MCTS_SIMULATIONS
    depth_turns: int = MCTS_DEPTH_TURNS
    pending_plan: Optional[Dict] = field(default=None, repr=False)

    def _rollout_policy(self, pname: str) -> HeuristicBot:
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
            try:
                g.roll_dice(active)
            except GameError:
                return score_total(g, myname)
            # white phase all
            for p in g.players:
                if g.white_phase_complete.get(p.name, False): continue
                bot = bots[p.name]
                if p.name == active:
                    bot.plan_turn(g); w_act = bot.pending_plan.get('white_action', ('skip', None))
                else:
                    w_act = bot.choose_white_sum_action(g)
                try:
                    self._apply_white(g, p.name, w_act)
                except GameError:
                    try: g.skip_white_sum(p.name)
                    except GameError: pass
            g.proceed_to_color_phase_if_ready()
            if g.phase == g.phase.COLOR_DICE:
                bot = bots[active]
                if bot.pending_plan is None: bot.plan_turn(g)
                c_act = bot.pending_plan.get('color_action', ('skip', None, None))
                try:
                    self._apply_color(g, active, c_act)
                except GameError:
                    marked_white = bool(g.turn_actions.get(active, {}).get('whiteSumMarked', False))
                    if not marked_white:
                        try: g.take_penalty(active)
                        except GameError: pass
            try:
                g.end_turn(active)
            except GameError:
                break
        return score_total(g, myname)

    def _evaluate_plan(self, g: QwixxGame, plan_w: Tuple[str, Optional[str]], plan_c: Tuple[str, Optional[str], Optional[int]]) -> float:
        base = score_total(g, self.name)
        sim = clone(g)
        active = sim.current_player
        try:
            self._apply_white(sim, active, plan_w)
        except GameError:
            return -1e9
        if sim.phase == sim.phase.WHITE_SUM:
            for p in sim.players:
                if not sim.white_phase_complete.get(p.name, False):
                    try: sim.skip_white_sum(p.name)
                    except GameError: pass
            sim.proceed_to_color_phase_if_ready()
        try:
            self._apply_color(sim, active, plan_c)
        except GameError:
            return -1e9
        try:
            sim.end_turn(active)
        except GameError:
            pass
        sims_per = max(1, self.simulations // max(1,1))
        total = 0.0
        for _ in range(sims_per):
            sim_i = clone(sim)
            val = self._simulate_future(sim_i, self.name, self.depth_turns)
            total += val
        return (total / sims_per) - base

    def plan_turn(self, g: QwixxGame):
        ws = g.white_sum()
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
            color_choices += [('mark', c, wd) for (c, wd, _) in acts for (_,__,___) in [(c,wd,0)] if True]
            for c_action in color_choices:
                if (not white_marked) and (c_action[0] == 'skip'): continue
                candidates.append(((w_action, w_color), c_action))

        best_pair = None
        best_value = -1e9
        sims_each = max(1, self.simulations // max(1, len(candidates)))
        for (w_act, c_act) in candidates:
            old = self.simulations
            self.simulations = sims_each
            val = self._evaluate_plan(g, w_act, c_act)
            self.simulations = old
            if val > best_value:
                best_value = val; best_pair = (w_act, c_act)
        if best_pair is None:
            hb = HeuristicBot(self.name); hb.plan_turn(g); self.pending_plan = hb.pending_plan; return
        self.pending_plan = {'white_action': best_pair[0], 'color_action': best_pair[1], 'value': best_value}

    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        hb = HeuristicBot(self.name); return hb.choose_white_sum_action(g)
    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan: return self.pending_plan.get('color_action', ('skip', None, None))
        hb = HeuristicBot(self.name); return hb.choose_color_action(g)

# ===== LearnerBot (NEW) =====
@dataclass
class LearnerBot:
    name: str
    epsilon: float = 0.10
    store_path: str = './qwixx_data/learner.json'
    prior_blend: float = 0.30  # weight for heuristic prior in selection
    pending_plan: Optional[Dict] = field(default=None, repr=False)
    _store: ExperienceStore = field(default_factory=ExperienceStore, init=False, repr=False)
    _hasher: FeatureHasher = field(default_factory=FeatureHasher, init=False, repr=False)
    _decisions: List[Tuple[str, str]] = field(default_factory=list, init=False, repr=False)  # (feature_key, action_key)

    def __post_init__(self):
        self._store.path = self.store_path
        self._store.load()

    # Fallbacks for non-active participants in white-sum
    def choose_white_sum_action(self, g: QwixxGame) -> Tuple[str, Optional[str]]:
        if self.pending_plan and self.pending_plan.get('white_action'):
            return self.pending_plan['white_action']
        # Use heuristic fallback when not planning whole turn
        hb = HeuristicBot(self.name)
        return hb.choose_white_sum_action(g)

    def choose_color_action(self, g: QwixxGame) -> Tuple[str, Optional[str], Optional[int]]:
        if self.pending_plan and self.pending_plan.get('color_action'):
            return self.pending_plan['color_action']
        hb = HeuristicBot(self.name)
        return hb.choose_color_action(g)

    def _evaluate_pair_with_prior(self, g: QwixxGame, feature_key: str,
                                  w_act: Tuple[str, Optional[str]],
                                  c_act: Tuple[str, Optional[str], Optional[int]]) -> float:
        # data-based expected reward
        akey = plan_key(w_act, c_act)
        q_est = self._store.expected(feature_key, akey, default=0.0)
        # small heuristic prior (use HeuristicBot planner score)
        hb = HeuristicBot(self.name)
        hb.plan_turn(g)  # we just want a scale reference; below we compute synthetic plan score quickly too
        # compute a fast heuristic value for this specific pair:
        # We approximate by temporarily applying pair on clone and computing immediate delta (no future rollouts).
        base = score_total(g, self.name)
        sim = clone(g)
        val = 0.0
        try:
            # Apply white
            if w_act[0] == 'mark' and w_act[1]:
                ws = sim.white_sum()
                sim.mark_number_white_sum(self.name, w_act[1], ws)
                sim.finalize_pending_row_closures()
            else:
                sim.skip_white_sum(self.name)
            # Move to color
            if sim.phase == sim.phase.WHITE_SUM:
                for p in sim.players:
                    if not sim.white_phase_complete.get(p.name, False):
                        try: sim.skip_white_sum(p.name)
                        except GameError: pass
                sim.proceed_to_color_phase_if_ready()
            # Apply color
            if c_act[0] == 'mark' and c_act[1] and c_act[2]:
                d = sim.dice
                number = (d['white1'] + d[c_act[1]]) if c_act[2] == 1 else (d['white2'] + d[c_act[1]])
                sim.select_dice_for_color_phase(self.name, c_act[2], c_act[1])
                sim.mark_number_color_dice(self.name, c_act[1], number)
            else:
                # penalty only if white skipped and color skipped (engine enforces at end_turn)
                pass
            val = score_total(sim, self.name) - base
        except GameError:
            val = -1e9
        # Combine
        return q_est + self.prior_blend * (val / 15.0)  # normalized a bit

    def plan_turn(self, g: QwixxGame):
        ws = g.white_sum()
        feature_key = self._hasher.hash_state(g, self.name)

        white_choices: List[Tuple[str, Optional[str]]] = [('skip', None)]
        white_choices += [('mark', c) for c in legal_white_sum_colors(g, self.name)]

        candidates: List[Tuple[Tuple[str,Optional[str]], Tuple[str,Optional[str],Optional[int]]]] = []
        # enumerate like in MCTS/Heuristic
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
            color_choices += [('mark', c, wd) for (c, wd, _) in acts for (_,__,___) in [(c,wd,0)] if True]
            for c_action in color_choices:
                if (not white_marked) and (c_action[0] == 'skip'): continue
                candidates.append(((w_action, w_color), c_action))

        # epsilon-greedy selection among candidates using store + heuristic prior
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
        # remember the (feature, action) taken to credit after game ends
        self._decisions.append((feature_key, plan_key(w_act, c_act)))
        self.pending_plan = {'white_action': w_act, 'color_action': c_act, 'value': score}

    # Called by UI/Autoplay when the game finishes
    def notify_game_end(self, final_scores: Dict[str, int], save: bool = True):
        reward = default_reward_from_scores(final_scores, self.name)
        for fk, ak in self._decisions:
            self._store.update(fk, ak, reward)
        self._decisions.clear()
        if save:
            self._store.save()
