#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heuristic tuning harness: random HEUR profiles → vs gap baselines → rank → mini RR among top-K.
Requires autoplay factory support for 'heur_profile:NAME' (see instructions in the file).
"""

import json, random
from pathlib import Path
from typing import Dict, List
from qwixx.autoplay import play_series
from qwixx import ai as ai_mod

OUTDIR = Path('./tuning_out'); OUTDIR.mkdir(parents=True, exist_ok=True)
BASELINES = ["gap1", "gap1lock", "gap2", "gap2lock"]

# ====== SEARCH SPACE (tweak ranges as you like) ======
RANGES = {
    'setup_bonus': (3.5, 6.0),
    'flex_weight': (0.12, 0.26),
    'big_skip_penalty_per_cell': (0.90, 1.30),
    'close_now_bonus': (8.0, 11.0),
    'no_lock_cap_penalty': (5.0, 8.0),
}
# Keep these tied to your current logic unless you want to explore them too:
FIXED = {
    'min_component_value': ai_mod.MIN_COMPONENT_VALUE,
    'combo_delta_margin': ai_mod.COMBO_DELTA_MARGIN,
}

# ====== TOURNAMENT SETTINGS ======
GAMES_PER_SERIES = 200     # use 200 for faster iteration; bump to 500+ to confirm winners
PROFILES_TO_TRY = 16       # number of random profiles per batch
TOP_K = 4                  # top profiles to RR among themselves
SEED_BASE = 9001
random.seed(SEED_BASE)

def sample_profile() -> Dict[str, float]:
    def rnd(a,b): return round(random.uniform(a,b), 2)
    p = {k: rnd(*RANGES[k]) for k in RANGES}
    p.update(FIXED)
    return p

def register_profile(name: str, params: Dict[str, float]):
    if not hasattr(ai_mod, 'HEUR_PROFILES'):
        ai_mod.HEUR_PROFILES = {}
    ai_mod.HEUR_PROFILES[name] = params

def run_vs_baselines(profile_name: str) -> List[Dict]:
    results = []
    for b in BASELINES:
        for order in ('AB','BA'):
            if order == 'AB':
                names = [f"A_{profile_name}", f"B_{b}"]
                kinds = [f"heur_profile:{profile_name}", b]
            else:
                names = [f"A_{b}", f"B_{profile_name}"]
                kinds = [b, f"heur_profile:{profile_name}"]
            bot_types = {names[0]: kinds[0], names[1]: kinds[1]}
            seed = SEED_BASE + hash((profile_name, b, order)) % 10000
            rep = play_series(
                names=names, bot_types=bot_types, games=GAMES_PER_SERIES,
                mcts_sims=0, mcts_depth=0, seed=seed, show_each=False,
                progress_interval=0.5, epsilon=0.10, epsilon_end=0.05, epsilon_decay=0.995,
                learn_file='./qwixx_data/learner.json', debug_timing=False
            )
            results.append({'baseline': b, 'order': order, 'report': rep})
    return results

def summarize(profile_name: str, series: List[Dict]) -> Dict:
    wins = 0; games = 0; score_sum = 0.0
    by_b = {}
    for rec in series:
        rep = rec['report']; g = rep['games']; games += g
        wins_map = rep['wins']; avgs = rep['avg_scores']
        if rec['order'] == 'AB':
            w = wins_map.get(f"A_{profile_name}", 0)
            s = avgs.get(f"A_{profile_name}", 0.0)
        else:
            w = wins_map.get(f"B_{profile_name}", 0)
            s = avgs.get(f"B_{profile_name}", 0.0)
        wins += w; score_sum += s * g
        b = rec['baseline']
        bb = by_b.setdefault(b, {'games':0, 'wins':0, 'avg_sum':0.0})
        bb['games'] += g; bb['wins'] += w; bb['avg_sum'] += s * g
    wr = wins / games if games else 0.0
    avg = score_sum / games if games else 0.0
    for b, v in by_b.items():
        v['avg'] = v['avg_sum'] / v['games'] if v['games'] else 0.0
        del v['avg_sum']
    return {'profile': profile_name, 'winrate': wr, 'avg': avg, 'by_baseline': by_b}

def main():
    # 1) sample and register random profiles
    profiles = {}
    for i in range(PROFILES_TO_TRY):
        name = f"rand{i+1:02d}"
        params = sample_profile()
        profiles[name] = params
        register_profile(name, params)
    (OUTDIR / 'profiles.json').write_text(json.dumps(profiles, indent=2))

    # 2) run vs baselines
    blocks = {}
    for name in profiles:
        print(f"== Running {name} vs baselines ==")
        blocks[name] = run_vs_baselines(name)
        (OUTDIR / f'{name}_block.json').write_text(json.dumps(blocks[name], indent=2))

    # 3) summarize and rank
    summary = [summarize(name, series) for name, series in blocks.items()]
    summary.sort(key=lambda x: (-x['winrate'], -x['avg']))
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2))

    print("\nTop profiles vs baselines:")
    for i,row in enumerate(summary[:TOP_K], start=1):
        print(f"{i:>2}. {row['profile']}: WR={row['winrate']*100:.1f}%  AVG={row['avg']:.2f}")

    # 4) optional mini round-robin among top-K HEURs
    top = [row['profile'] for row in summary[:TOP_K]]
    rr = []
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            A, B = top[i], top[j]
            for order in ('AB','BA'):
                if order == 'AB':
                    names = [f"A_{A}", f"B_{B}"]
                    kinds = [f"heur_profile:{A}", f"heur_profile:{B}"]
                else:
                    names = [f"A_{B}", f"B_{A}"]
                    kinds = [f"heur_profile:{B}", f"heur_profile:{A}"]
                bot_types = {names[0]: kinds[0], names[1]: kinds[1]}
                seed = SEED_BASE + hash((A,B,order)) % 10000
                rep = play_series(
                    names=names, bot_types=bot_types, games=GAMES_PER_SERIES,
                    mcts_sims=0, mcts_depth=0, seed=seed, show_each=False,
                    progress_interval=0.5, epsilon=0.10, epsilon_end=0.05, epsilon_decay=0.995,
                    learn_file='./qwixx_data/learner.json', debug_timing=False
                )
                rr.append({'A':A, 'B':B, 'order':order, 'report':rep})
    (OUTDIR / 'round_robin.json').write_text(json.dumps(rr, indent=2))
    print("Saved:", OUTDIR)

if __name__ == '__main__':
    main()
