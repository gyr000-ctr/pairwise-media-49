"""贝塞尔三次曲线的弧长计算与弧长反解。

算法
----
三次贝塞尔曲线的速度 ``v(t) = q0 + q1 t + q2 t^2`` 是二次向量多项式，
速率 ``|v(t)|`` 的平方是四次多项式。弧长

    L(t) = ∫_0^t |v(u)| du

在 ``[0, 1]`` 上单调不减。

* **分段**：在 ``[0, 1]`` 上先按均匀面板切开，并在曲线尖点
  （``|v(t)| = 0`` 的内部点）处额外切开。尖点两侧被积函数解析，数值求积
  可达到机器精度。
* **求积**：每个子区间用 Gauss-Kronrod 7-15 一次得到积分值与本地点误差
  估计；不达标的子区间二分递归（深度有上限），叶节点构成分段弧长表。
  每次求值只遍历一次节点，同时累积 15 点（Kronrod）与 7 点（Gauss）结果。
* **反解**：在分段弧长表上二分定位区间，再以 ``L'(t) = |v(t)|`` 做带区间
  保护的牛顿迭代（越界或速率为零时退化为二分），保证单调收敛。反解与建表
  使用同一套求积规则，二者一致，残差可达机器精度。
* **并发**：曲线表构建后只读；``BezierSplit`` 实例用锁保护的 LRU 缓存
  保存最近使用的曲线表，可跨线程共享。反解是只读表上的纯计算。
* **异常**：非有限坐标 / 非有限弧长 / 维数不一致 / 弧长为零的退化曲线
  → ``ValueError``；非实数输入 → ``TypeError``。

本模块只依赖标准库。
"""

import bisect
import math
import threading
from collections import OrderedDict
from numbers import Real

__all__ = ["BezierSplit"]

# --------------------------------------------------------------------------
# Gauss-Kronrod 7-15 常数（在 [-1, 1] 上）
# --------------------------------------------------------------------------
# _KX / _KW：正半轴的 7 个非零节点及其 Kronrod 权重。
# _GW：这 15 个节点中 7 个 Gauss 节点（正半轴 1、3、5 号）的权重。
# _KW_CENTER / _GW_CENTER：零点节点的 Kronrod / Gauss 权重。

_KX = (0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
       0.7415311855993945, 0.5860872354676911, 0.4058451513773972,
       0.2077849550078985)
_KW = (0.02293532201052922, 0.06309209262997855, 0.10479001032225018,
       0.14065325971552592, 0.1690047266392679, 0.19035057806478542,
       0.20443294007529889)
_GW = {1: 0.1294849661688697, 3: 0.27970539148927667,
       5: 0.38183005050511894}
_KW_CENTER = 0.20948214108472783
_GW_CENTER = 0.4179591836734694


def _build_kronrod_table():
    nodes = []
    for i in range(6, -1, -1):
        nodes.append((-_KX[i], _KW[i], _GW.get(i, 0.0)))
    nodes.append((0.0, _KW_CENTER, _GW_CENTER))
    for i in range(7):
        nodes.append((_KX[i], _KW[i], _GW.get(i, 0.0)))
    return tuple(nodes)


_KRONROD = _build_kronrod_table()

_SQRT = math.sqrt

# 构建与反解参数
_PANELS = 2            # 初始均匀面板数（尖点会额外插入断点）
_REL_TOL = 1e-8        # 每个子区间的相对误差目标
_ABS_TOL = 1e-15       # 绝对误差下限（相对控制多边形长度）
_MAX_DEPTH = 14        # 自适应二分最大深度
_SOLVE_TOL = 1e-12     # 反解弧长残差目标（相对总弧长）
_SOLVE_STEPS = 60      # 反解迭代上限
_CACHE_MAX = 64        # 每个实例缓存的曲线表数量


# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------


def _finite_float(value, what):
    """把输入转换为有限浮点数；非实数 -> TypeError，非有限 -> ValueError。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("%s 必须是实数" % what)
    try:
        out = float(value)
    except (OverflowError, ValueError) as exc:  # pragma: no cover - 极端输入
        raise ValueError("%s 无法转换为浮点数" % what) from exc
    if not math.isfinite(out):
        raise ValueError("%s 必须是有限数" % what)
    return out


def _coordinate(value, what):
    """坐标快速校验：float 走快路径，其余交给 _finite_float 分类处理。"""
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("%s 必须是有限数" % what)
        return value
    if type(value) is int:
        try:
            return float(value)
        except OverflowError:
            raise ValueError("%s 超出浮点范围" % what) from None
    return _finite_float(value, what)


def _prepare(p0, p1, p2, p3):
    """校验并规范化四个控制点，返回 ((x, y, ...), ...)。"""
    points = []
    dims = None
    for idx, point in enumerate((p0, p1, p2, p3)):
        try:
            values = tuple(point)
        except TypeError:
            raise TypeError("控制点 p%d 必须是实数序列" % idx) from None
        coords = tuple(_coordinate(v, "p%d 的坐标" % idx) for v in values)
        if dims is None:
            dims = len(coords)
        elif len(coords) != dims:
            raise ValueError("所有控制点的维数必须一致")
        points.append(coords)
    if not dims:
        raise ValueError("控制点不能为空")
    return tuple(points)


# --------------------------------------------------------------------------
# 数值工具
# --------------------------------------------------------------------------


def _quadratic_roots(a, b, c):
    """稳定求解 a*t^2 + b*t + c = 0 的实根。"""
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0.0:
        return ()
    eps = 1e-14 * scale
    if abs(a) <= eps:  # 退化为线性
        if abs(b) <= eps:
            return ()
        return (-c / b,)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        if disc < -1e-13 * (b * b + abs(4.0 * a * c)):
            return ()
        disc = 0.0
    sq = math.sqrt(disc)
    q = -0.5 * (b + math.copysign(sq, b))
    if q == 0.0:
        return (0.0,)
    return (q / a, c / q)


def _find_cusps(velocity):
    """找出 (0, 1) 内速度为零的尖点参数。

    尖点处每个分量的速度都为 0，故任一非零分量的根都是尖点的超集；
    取第一个非零分量的根作候选，再用完整速率筛选。
    """
    driver = None
    for q0, q1, q2 in velocity:
        if q0 != 0.0 or q1 != 0.0 or q2 != 0.0:
            driver = (q0, q1, q2)
            break
    if driver is None:
        return ()

    candidates = [r for r in _quadratic_roots(driver[2], driver[1], driver[0])
                  if 1e-12 < r < 1.0 - 1e-12]
    if not candidates:
        return ()

    vscale = math.sqrt(sum((abs(q0) + abs(q1) + abs(q2)) ** 2
                           for q0, q1, q2 in velocity))
    tol = (1e-9 * vscale) ** 2

    def speed_sq(t):
        total = 0.0
        for q0, q1, q2 in velocity:
            v = (q2 * t + q1) * t + q0
            total += v * v
        return total

    cusps = []
    for r in sorted(candidates):
        if speed_sq(r) > tol:
            continue
        # 用导数绝对值最大的分量做一步 Newton 精化
        best_deriv = 0.0
        best_val = 0.0
        for q0, q1, q2 in velocity:
            val = (q2 * r + q1) * r + q0
            deriv = 2.0 * q2 * r + q1
            if abs(deriv) > abs(best_deriv):
                best_deriv = deriv
                best_val = val
        if best_deriv != 0.0:
            r -= best_val / best_deriv
        if not (1e-12 < r < 1.0 - 1e-12):
            continue
        if cusps and abs(r - cusps[-1]) <= 1e-9:
            continue
        cusps.append(r)
    return tuple(cusps)


# --------------------------------------------------------------------------
# 速率与求积（按维数分派）
# --------------------------------------------------------------------------


def _kronrod2(velocity, a, b):
    (q0x, q1x, q2x), (q0y, q1y, q2y) = velocity
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    k = 0.0
    g = 0.0
    for x, wk, wg in _KRONROD:
        t = mid + half * x
        vx = (q2x * t + q1x) * t + q0x
        vy = (q2y * t + q1y) * t + q0y
        s = _SQRT(vx * vx + vy * vy)
        k += wk * s
        if wg:
            g += wg * s
    return k * half, g * half


def _kronrod3(velocity, a, b):
    (q0x, q1x, q2x), (q0y, q1y, q2y), (q0z, q1z, q2z) = velocity
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    k = 0.0
    g = 0.0
    for x, wk, wg in _KRONROD:
        t = mid + half * x
        vx = (q2x * t + q1x) * t + q0x
        vy = (q2y * t + q1y) * t + q0y
        vz = (q2z * t + q1z) * t + q0z
        s = _SQRT(vx * vx + vy * vy + vz * vz)
        k += wk * s
        if wg:
            g += wg * s
    return k * half, g * half


def _kronrodn(velocity, a, b):
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    k = 0.0
    g = 0.0
    for x, wk, wg in _KRONROD:
        t = mid + half * x
        total = 0.0
        for q0, q1, q2 in velocity:
            v = (q2 * t + q1) * t + q0
            total += v * v
        s = _SQRT(total)
        k += wk * s
        if wg:
            g += wg * s
    return k * half, g * half


def _speed2(velocity, t):
    (q0x, q1x, q2x), (q0y, q1y, q2y) = velocity
    vx = (q2x * t + q1x) * t + q0x
    vy = (q2y * t + q1y) * t + q0y
    return _SQRT(vx * vx + vy * vy)


def _speed3(velocity, t):
    (q0x, q1x, q2x), (q0y, q1y, q2y), (q0z, q1z, q2z) = velocity
    vx = (q2x * t + q1x) * t + q0x
    vy = (q2y * t + q1y) * t + q0y
    vz = (q2z * t + q1z) * t + q0z
    return _SQRT(vx * vx + vy * vy + vz * vz)


def _speedn(velocity, t):
    total = 0.0
    for q0, q1, q2 in velocity:
        v = (q2 * t + q1) * t + q0
        total += v * v
    return _SQRT(total)


def _select_dims(dims):
    if dims == 2:
        return _kronrod2, _speed2
    if dims == 3:
        return _kronrod3, _speed3
    return _kronrodn, _speedn


def _velocity(points):
    """返回每个维度的速度二次系数 (q0, q1, q2)。

    系数按差分组计算，避免 ``p3 - 3 p2 + 3 p1 - p0`` 这类表达式在控制点
    重合时的浮点抵消误差（重合点应得到精确的零速率，从而被识别为退化曲线）。
    """
    out = []
    for d in range(len(points[0])):
        p0, p1, p2, p3 = (p[d] for p in points)
        q0 = 3.0 * (p1 - p0)
        q1 = 6.0 * ((p2 - p1) - (p1 - p0))
        q2 = 3.0 * ((p3 - p0) - 3.0 * (p2 - p1))
        out.append((q0, q1, q2))
    return tuple(out)


# --------------------------------------------------------------------------
# 曲线弧长表（构建后只读，可跨线程共享）
# --------------------------------------------------------------------------


class _ArcTable:
    """一条三次贝塞尔曲线的分段弧长表。

    内部在归一化坐标下积分：先平移到以 p0 为原点、再按包围盒尺度缩放。
    这样既消除了远离原点时 ``p1 - p0`` 的相消误差，也避免了速度平方对量级
    的敏感（过小下溢为零、过大溢出）。``t`` 与平移、正比例缩放无关，弧长
    先按 ``scale`` 还原，再整体平移不影响弧长。
    """

    __slots__ = ("velocity", "total", "ts", "cums", "_kq", "_speed",
                 "_scale", "_tnorm")

    def __init__(self, points):
        dims = len(points[0])

        # 平移到 p0：消除大坐标偏移带来的相消误差
        ox = points[0]
        shifted = tuple(tuple(pt[d] - ox[d] for d in range(dims))
                        for pt in points)

        # 按包围盒尺度缩放：使坐标量级约为 1
        scale = 0.0
        for pt in shifted:
            for c in pt:
                a = abs(c)
                if a > scale:
                    scale = a
        if scale > 0.0:
            points = tuple(tuple(c / scale for c in pt) for pt in shifted)
        else:
            scale = 1.0
        self._scale = scale

        velocity = _velocity(points)
        kq, speed = _select_dims(dims)
        self.velocity = velocity
        self._kq = kq
        self._speed = speed

        # 控制多边形长度：真实弧长的上界，用作误差的绝对尺度
        poly = 0.0
        for i in range(3):
            poly += math.hypot(*(points[i + 1][d] - points[i][d]
                                 for d in range(dims)))
        panel_floor = _ABS_TOL * poly / _PANELS

        # 断点：均匀面板 + 尖点
        breaks = [i / _PANELS for i in range(1, _PANELS)]
        breaks.extend(_find_cusps(velocity))

        ts = [0.0]
        cums = [0.0]
        acc = [0.0]
        prev = 0.0
        for br in sorted(breaks):
            if br - prev > 1e-12:
                self._refine(kq, velocity, prev, br, 0, panel_floor, ts, cums, acc)
                prev = br
        self._refine(kq, velocity, prev, 1.0, 0, panel_floor, ts, cums, acc)
        if ts[-1] != 1.0:
            ts[-1] = 1.0

        self.ts = tuple(ts)
        self.cums = tuple(cums)
        self._tnorm = acc[0]
        self.total = acc[0] * scale

    @staticmethod
    def _refine(kq, velocity, a, b, depth, floor, ts, cums, acc):
        """对 [a, b] 自适应求积，把叶节点边界与累计弧长写入 ts/cums。"""
        k, g = kq(velocity, a, b)
        if depth >= _MAX_DEPTH or abs(k - g) <= _REL_TOL * k + floor:
            if ts[-1] != b:
                ts.append(b)
            acc[0] += k
            cums.append(acc[0])
            return
        mid = 0.5 * (a + b)
        half_floor = floor * 0.5
        _ArcTable._refine(kq, velocity, a, mid, depth + 1, half_floor,
                          ts, cums, acc)
        _ArcTable._refine(kq, velocity, mid, b, depth + 1, half_floor,
                          ts, cums, acc)

    # -- 反解 ---------------------------------------------------------------

    def solve(self, s):
        """求 t 使（归一化后的）弧长为 s（调用方保证 0 < s < total_norm）。"""
        cums = self.cums
        ts = self.ts
        kq = self._kq
        speed = self._speed
        velocity = self.velocity

        i = bisect.bisect_right(cums, s) - 1
        if i < 0:
            i = 0
        elif i >= len(cums) - 1:
            i = len(cums) - 2

        start = ts[i]
        end = ts[i + 1]
        base = cums[i]
        span = cums[i + 1] - base
        frac = (s - base) / span if span > 0.0 else 0.0
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0

        t = start + (end - start) * frac
        lo = start
        hi = end
        tol = _SOLVE_TOL * self._tnorm

        for _ in range(_SOLVE_STEPS):
            k, _g = kq(velocity, start, t)
            err = base + k - s
            if abs(err) <= tol:
                break
            v = speed(velocity, t)
            if v > 0.0:
                candidate = t - err / v
                if lo < candidate < hi:
                    if err > 0.0:
                        hi = t
                    else:
                        lo = t
                    t = candidate
                    continue
            # 牛顿步越界或速率为零：二分兜底
            if err > 0.0:
                hi = t
            else:
                lo = t
            mid = 0.5 * (lo + hi)
            if mid == t:
                break
            t = mid

        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return float(t)


# --------------------------------------------------------------------------
# 对外 API
# --------------------------------------------------------------------------


class BezierSplit:
    """三次贝塞尔曲线的弧长与弧长反解。

    实例可在多线程间共享：曲线表构建后只读，缓存访问加锁，反解是纯计算。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cache = OrderedDict()

    def _table(self, p0, p1, p2, p3):
        key = _prepare(p0, p1, p2, p3)
        with self._lock:
            table = self._cache.get(key)
            if table is not None:
                self._cache.move_to_end(key)
                return table
        # 构建在锁外进行：同一控制点并发重复构建无副作用
        table = _ArcTable(key)
        if not (table._tnorm > 0.0):
            raise ValueError("退化曲线：弧长为零")
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return existing
            self._cache[key] = table
            while len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)
        return table

    def length(self, p0, p1, p2, p3):
        """返回曲线弧长。"""
        return self._table(p0, p1, p2, p3).total

    def t_at(self, p0, p1, p2, p3, s):
        """给定弧长 ``s`` 返回参数 ``t``（``0.0 <= t <= 1.0``）。"""
        s = _finite_float(s, "弧长 s")
        table = self._table(p0, p1, p2, p3)
        total = table.total
        slack = 1e-12 * total
        if s < -slack or s > total + slack:
            raise ValueError("弧长 s 超出 [0, length] 范围")
        if s <= 0.0:
            return 0.0
        if s >= total:
            return 1.0
        # 表内弧长以归一化坐标计，按 scale 换回
        return table.solve(s / table._scale)
