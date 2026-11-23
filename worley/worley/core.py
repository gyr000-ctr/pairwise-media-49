"""可平铺 Worley 噪声（返回 F1/F2）。

- ``sample(x, y)`` 返回 ``(F1, F2)``，恒有 ``0 <= F1 <= F2``；
- 任意位置（含格边界、负坐标）均扫描完整 3x3 邻域，不越界、不错格；
- ``tile`` 可配：指定每周期的格数后噪声按该周期平铺，
  接缝处距离场连续（特征点按格索引取模重复，跳变为 0 量级）；
- 同种子完全可复现；``sample`` 不持有任何可变状态，天然并发安全。
"""

import math

__all__ = ["Worley"]

_MASK = 0xFFFFFFFFFFFFFFFF
_P1 = 0x9E3779B97F4A7C15
_P2 = 0xC2B2AE3D27D4EB4F
_M1 = 0xBF58476D1CE4E5B9
_INV32 = 1.0 / 4294967296.0  # 2**-32
_INF = float("inf")
_floor = math.floor
_sqrt = math.sqrt


def _as_seed(seed):
    """校验种子：非负整数。类型错误与取值错误可区分。"""
    if isinstance(seed, bool):
        raise TypeError("seed 必须为整数，不能是 bool")
    if isinstance(seed, float):
        if not math.isfinite(seed):
            raise ValueError("seed 必须为有限值")
        if not seed.is_integer():
            raise TypeError("seed 必须为整数")
        seed = int(seed)
    if not isinstance(seed, int):
        raise TypeError("seed 必须为整数")
    if seed < 0:
        raise ValueError("seed 必须为非负整数")
    return seed


def _as_cell(cell):
    """校验格宽：有限正实数。"""
    if isinstance(cell, bool) or not isinstance(cell, (int, float)):
        raise TypeError("cell 必须为实数")
    cell = float(cell)
    if not math.isfinite(cell):
        raise ValueError("cell 必须为有限值")
    if cell <= 0.0:
        raise ValueError("cell 必须为正数")
    return cell


def _as_tile(tile):
    """校验平铺尺寸：None 或 >=1 的整数（每周期的格数）。"""
    if tile is None:
        return None
    if isinstance(tile, bool):
        raise TypeError("tile 必须为整数格数，不能是 bool")
    if isinstance(tile, float):
        if not math.isfinite(tile):
            raise ValueError("tile 必须为有限值")
        if not tile.is_integer():
            raise TypeError("tile 必须为整数格数")
        tile = int(tile)
    if not isinstance(tile, int):
        raise TypeError("tile 必须为整数格数")
    if tile < 1:
        raise ValueError("tile 必须 >= 1")
    return tile


class Worley:
    """可平铺 Worley 噪声采样器。"""

    def __init__(self, seed=0, cell=1.0, tile=None):
        """构造。

        :param seed: 非负整数种子，同种子结果可复现；
        :param cell: 格宽（正有限实数），F1/F2 以世界单位返回；
        :param tile: 每周期的格数（>=1 的整数）或 None（不平铺）。
        """
        self._seed = _as_seed(seed)
        self._cell = _as_cell(cell)
        self._tile = _as_tile(tile)
        self._inv_cell = 1.0 / self._cell
        # 种子混合常数（乘奇数常数 mod 2**64 是双射，异种子必异状态）
        self._sz = (self._seed * _P1 + _P2) & _MASK

    def sample(self, x, y):
        """返回 (F1, F2)：到最近与次近特征点的距离，F1 <= F2。"""
        if type(x) is not float or type(y) is not float:
            if (isinstance(x, bool) or not isinstance(x, (int, float))
                    or isinstance(y, bool) or not isinstance(y, (int, float))):
                raise TypeError("采样坐标必须为实数")
            x = float(x)
            y = float(y)
        if not (-_INF < x < _INF and -_INF < y < _INF):
            raise ValueError("采样坐标必须为有限实数")
        inv = self._inv_cell
        u = x * inv
        v = y * inv
        ci = _floor(u)
        cj = _floor(v)
        i0 = ci - 1
        j0 = cj - 1
        p1 = _P1
        p2 = _P2
        mask = _MASK
        t = self._tile
        if t is None:
            ris = ((i0 * p1) & mask, ((i0 + 1) * p1) & mask, ((i0 + 2) * p1) & mask)
            rjs = ((j0 * p2) & mask, ((j0 + 1) * p2) & mask, ((j0 + 2) * p2) & mask)
        else:
            # 特征点按格索引取模重复 -> 场严格周期化，接缝连续
            ris = (((i0 % t) * p1) & mask, (((i0 + 1) % t) * p1) & mask,
                   (((i0 + 2) % t) * p1) & mask)
            rjs = (((j0 % t) * p2) & mask, (((j0 + 1) % t) * p2) & mask,
                   (((j0 + 2) % t) * p2) & mask)
        ius = (i0 - u, ci - u, ci + 1 - u)
        jvs = (j0 - v, cj - v, cj + 1 - v)
        sz = self._sz
        m1 = _M1
        inv32 = _INV32
        d1 = _INF
        d2 = _INF
        for k in (0, 1, 2):
            hb = ris[k] ^ sz
            iu = ius[k]
            for n in (0, 1, 2):
                # 确定性整数混合（乘奇数常数 + 高低位折叠）
                h = ((hb ^ rjs[n]) * m1) & mask
                h ^= h >> 32
                dx = iu + (h & 0xFFFFFFFF) * inv32
                dy = jvs[n] + (h >> 32) * inv32
                d = dx * dx + dy * dy
                if d < d1:
                    d2 = d1
                    d1 = d
                elif d < d2:
                    d2 = d
        cell = self._cell
        return (_sqrt(d1) * cell, _sqrt(d2) * cell)
