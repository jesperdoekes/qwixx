# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import random, time, sys

from .game import QwixxGame, Player, GameError, GameStatus, Phase
from .ai import (
    HeuristicBot, MCTSBot, LearnerBot, legal_color_actions,
    Gap1Bot, Gap1LockBot, Gap2Bot, Gap2LockBot
)

def _apply_white_for_all(g: QwixxGame, bots: Dict[str, object]):
    active = g.current_player
    if active in bots and getattr(bots[active], 'pending_plan', None) is None:
        if hasattr(bots[active], 'plan_turn'):
            bots[active].plan_turn(g)

    for p in g.players:
        if g.white_phase_complete.get(p.name, False):
            continue
        bot = bots[p.name]
        if p.name == active and getattr(bot, 'pending_plan', None):
            action, color = bot.pending_plan.get('white_action', ('skip', None))
        else:
            action, color = bot.choose_white_sum_action(g)
        try:
            if action == 'mark' and color:
                ws = g.white_sum()
                g.mark_number_white_sum(p.name, color, ws)
            else:
                g.skip_white_sum(p.name)
        except GameError:
            try: g.skip_white_sum(p.name)
            except GameError: pass

    g.proceed_to_color_phase_if_ready()

def _apply_color_for_active(g: QwixxGame, bots: Dict[str, object]):
    active = g.current_player
    bot = bots[active]
    marked_white = bool(g.turn_actions.get(active, {}).get('whiteSumMarked', False))

    if getattr(bot, 'pending_plan', None) is None and hasattr(bot, 'plan_turn'):
        bot.plan_turn(g)

    act, color, wd = bot.pending_plan.get('color_action', ('skip', None, None)) if getattr(bot, 'pending_plan', None) else ('skip', None, None)
    try:
        if act == 'mark' and color and wd:
            g.select_dice_for_color_phase(active, wd, color)
            cv = g.combined_value()
            g.mark_number_color_dice(active, color, cv)
        else:
            if not marked_white:
                g.take_penalty(active)
    except GameError:
        acts = legal_color_actions(g, active)
        try:
            if acts:
                c2, wd2, n2 = acts[0]
                g.select_dice_for_color_phase(active, wd2, c2)
                g.mark_number_color_dice(active, c2, n2)
            elif not marked_white:
                g.take_penalty(active)
        except GameError:
            pass
    finally:
        # Learners get immediate per-turn reward
        if isinstance(bot, LearnerBot):
            bot.notify_step_outcome(g)
        if hasattr(bot, 'pending_plan'):
            bot.pending_plan = None

def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def play_one_game(names: List[str],
                  bot_types: Dict[str, str],
                  mcts_sims: int = 400,
                  mcts_depth: int = 2,
                  start_index: int = 0,
                  seed: Optional[int] = None,
                  epsilon: float = 0.10,
                  learn_file: Optional[str] = None) -> Dict:
    # Use per-game seed; also seed global PRNG because QwixxGame.roll_dice uses random.*
    rnd = random.Random(seed) if seed is not None else random
    if seed is not None:
        random.seed(seed)

    players = [Player(n, is_host=(i==0)) for i, n in enumerate(names)]
    g = QwixxGame(game_id='auto', players=players, current_player=names[start_index])
    g.select_first_player(names[start_index])

    bots: Dict[str, object] = {}
    for n in names:
        kind = bot_types.get(n, 'heur').lower()
        if kind == 'mcts':
            bots[n] = MCTSBot(n, simulations=mcts_sims, depth_turns=mcts_depth)
        elif kind == 'learn':
            bots[n] = LearnerBot(n, epsilon=epsilon, store_path=(learn_file or './qwixx_data/learner.json'))
        elif kind == 'gap1':
            bots[n] = Gap1Bot(n)
        elif kind == 'gap1lock':
            bots[n] = Gap1LockBot(n)
        elif kind == 'gap2':
            bots[n] = Gap2Bot(n)
        elif kind == 'gap2lock':
            bots[n] = Gap2LockBot(n)
        else:
            bots[n] = HeuristicBot(n)

    guard_turns = 0
    while g.status != GameStatus.FINISHED:
        active = g.current_player
        try:
            g.roll_dice(active)
        except GameError:
            break
        _apply_white_for_all(g, bots)
        if g.phase == Phase.COLOR_DICE:
            _apply_color_for_active(g, bots)
        try:
            g.end_turn(active)
        except GameError:
            break
        guard_turns += 1
        if guard_turns > 500:
            break

    scores = {p.name: g.calculate_score(p.name)['total'] for p in g.players}
    # notify learners (final reward)
    for n, bot in bots.items():
        if isinstance(bot, LearnerBot):
            bot.notify_game_end(scores, save=True)

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = sorted_scores[0][1]
    winners = [name for name, sc in sorted_scores if sc == top]
    second = sorted_scores[1][1] if len(sorted_scores) > 1 else top
    margin = top - second
    return {'scores': scores, 'winners': winners, 'margin': margin}

def play_series(names: List[str],
                bot_types: Dict[str, str],
                games: int,
                mcts_sims: int = 400,
                mcts_depth: int = 2,
                seed: Optional[int] = None,
                show_each: bool = False,
                progress_interval: float = 0.5,
                epsilon: float = 0.10,
                epsilon_end: float = 0.05,
                epsilon_decay: float = 0.995,
                learn_file: Optional[str] = None,
                debug_timing: bool = False) -> Dict:
    base_rng = random.Random(seed) if seed is not None else random
    wins = {n: 0 for n in names}
    ties = 0
    margins: List[int] = []
    score_sums = {n: 0 for n in names}
    t0 = time.time()
    last_tick = 0.0

    def tick(done: int):
        nonlocal last_tick
        now = time.time()
        if (now - last_tick) < progress_interval and done != games: return
        last_tick = now
        elapsed = now - t0
        avg = elapsed / done if done > 0 else 0.0
        remain = (games - done) * avg
        pct = int(100 * (done / games)) if games > 0 else 100
        sys.stdout.write("\r" + f"Game {done}/{games} ({pct}%) Elapsed {_fmt_hms(elapsed)} ETA {_fmt_hms(remain)} ")
        sys.stdout.flush()
        if done == games: sys.stdout.write("\n")

    tick(0)
    for gi in range(games):
        start_index = gi % len(names)
        per_game_seed = base_rng.randint(0, 2**31 - 1) if seed is not None else None
        # Schedule epsilon (for learners)
        eps_now = max(epsilon_end, epsilon * (epsilon_decay ** gi))
        t_start = time.time()
        res = play_one_game(
            names=names,
            bot_types=bot_types,
            mcts_sims=mcts_sims,
            mcts_depth=mcts_depth,
            start_index=start_index,
            seed=per_game_seed,
            epsilon=eps_now,
            learn_file=learn_file
        )
        t_end = time.time()
        for n in names:
            score_sums[n] += res['scores'][n]
        margins.append(res['margin'])

        if len(res['winners']) == 1:
            wins[res['winners'][0]] += 1
            if show_each:
                print(f"\nGame {gi+1}: winner={res['winners'][0]} scores={res['scores']} margin={res['margin']}")
        else:
            ties += 1
            if show_each:
                print(f"\nGame {gi+1}: TIE winners={res['winners']} scores={res['scores']}")

        if debug_timing:
            print(f"[Timing] Game {gi+1} took {(t_end - t_start):.3f}s")

        tick(gi + 1)

    avg_scores = {n: (score_sums[n] / games if games > 0 else 0.0) for n in names}
    avg_margin = (sum(margins) / len(margins)) if margins else 0.0
    return {'games': games, 'wins': wins, 'ties': ties, 'avg_scores': avg_scores, 'avg_margin': avg_margin, 'margins': margins}
