# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import json, os, math
from .game import QwixxGame, Player, COLORS

DEFAULT_STORE_PATH = os.path.join('.', 'qwixx_data', 'learner.json')

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

@dataclass
class ExperienceStore:
    path: str = DEFAULT_STORE_PATH
    data: Dict[str, Dict[str, Tuple[int, float]]] = field(default_factory=dict)
    # data[feature_key][action_key] = (count, sum_reward)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # Convert lists to tuples
            self.data = {fk: {ak: (int(v[0]), float(v[1])) for ak, v in d.items()} for fk, d in raw.items()}

    def save(self):
        _ensure_dir(self.path)
        raw = {fk: {ak: [int(cnt), float(s)] for ak, (cnt, s) in d.items()} for fk, d in self.data.items()}
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(raw, f, ensure_ascii=False, indent=2, sort_keys=True)

    def expected(self, feature_key: str, action_key: str, default: float = 0.0) -> float:
        d = self.data.get(feature_key, {})
        cnt, s = d.get(action_key, (0, 0.0))
        if cnt <= 0: return default
        return s / cnt

    def update(self, feature_key: str, action_key: str, reward: float):
        d = self.data.setdefault(feature_key, {})
        cnt, s = d.get(action_key, (0, 0.0))
        d[action_key] = (cnt + 1, s + float(reward))

@dataclass
class FeatureHasher:
    """
    Build a compact, stable feature key for 'turn state' prior to selecting a turn plan.
    """
    def row_frontier(self, p: Player, color: str) -> int:
        row = getattr(p, color)
        if not row:
            return -1  # empty
        if color in ('red','yellow'):
            return max(row)
        else:
            return min(row)

    def frontier_bin(self, color: str, v: int) -> str:
        if v < 0:
            return 'E'  # empty
        if color in ('red','yellow'):  # ascending
            if v <= 4: return 'A'
            if v <= 8: return 'B'
            if v <= 11: return 'C'
            return 'D'  # 12
        else:  # descending
            if v >= 9: return 'A'
            if v >= 5: return 'B'
            if v >= 3: return 'C'
            return 'D'  # 2

    def hash_state(self, g: QwixxGame, player_name: str) -> str:
        p = next(pp for pp in g.players if pp.name == player_name)
        parts: List[str] = []
        # Dice (coarse)
        d = g.dice
        w1 = d.get('white1') or 0
        w2 = d.get('white2') or 0
        parts.append(f"W:{w1},{w2}")
        # Frontiers + counts
        for c in COLORS:
            f = self.row_frontier(p, c)
            fb = self.frontier_bin(c, f)
            cnt = len(getattr(p, c))
            closed = int(g.closed_rows.get(c, False))
            parts.append(f"{c[0]}:{fb}:{cnt}:{closed}")
        # Penalties
        parts.append(f"P:{p.penalties}")
        # Active dice mask (after closures)
        adm = ''.join('1' if g.active_dice.get(c, True) else '0' for c in COLORS)
        parts.append(f"A:{adm}")
        return '|'.join(parts)

def plan_key(white_action: Tuple[str, Optional[str]], color_action: Tuple[str, Optional[str], Optional[int]]) -> str:
    w_act, w_col = white_action
    c_act, c_col, c_wd = color_action
    return f"W:{w_act}:{w_col or '-'}|C:{c_act}:{c_col or '-'}:{c_wd if c_wd else '-'}"

def default_reward_from_scores(scores: Dict[str, int], me: str) -> float:
    my = scores[me]
    others = [v for k, v in scores.items() if k != me]
    best_opp = max(others) if others else my
    margin = my - best_opp
    # scale to [-1, 1] roughly
    val = max(-1.0, min(1.0, margin / 100.0))
    return val
