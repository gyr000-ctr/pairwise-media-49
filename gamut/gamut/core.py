"""gamut 核心 API：把任意 sRGB 颜色映射进 sRGB 色域。

映射规则
--------
1. 输入是三分量颜色 ``(r, g, b)``，分量按 sRGB 编码值给出（期望在 [0, 1]）。
2. 已经在色域内（三分量均在 [0, 1]）的颜色原样返回，不做任何舍入。
3. 越界颜色转到 OKLCh 后，在 **保持 OKLab 亮度 L 与色相 h 不变** 的
   前提下，把色度 C 收缩到 sRGB 色域边界。因此映射结果的亮度与输入
   完全一致，色相误差只来自浮点运算（远小于 2°）。
4. 亮度本身越界时（``L < 0``，或 ``L > 1`` 且颜色为中性色）没有满足
   "亮度不变" 的合法输出，抛 :class:`LightnessOutOfRangeError`。
   彩色且 ``L > 1`` 时 sRGB 中不存在该亮度，只能把亮度压到 1 再收缩
   色度（等价于映射到白色方向），以保证总能返回 [0, 1] 内的结果。

色域边界求法
------------
固定 (L, h) 后，线性 sRGB 的每个通道都是色度 C 的三次多项式
（OKLab→LMS 线性、LMS→线性 RGB 立方）。色域边界即这些多项式首次
离开 [0, 1] 的位置：单调通道用带括号保护的 Newton 求根，非单调通道
用 Cardano 闭式根切分区间后逐段判定。单色开销是常数级，不做扫描。

并发安全
--------
:class:`GamutMap` 不持有任何可变状态；映射函数是纯函数，并带线程安全的
有界 LRU 缓存（``functools.lru_cache`` 自带锁）。多线程并发调用安全、
结果确定。
"""

import functools
import math

__all__ = [
    "GamutMap",
    "GamutError",
    "ColorFormatError",
    "NonFiniteColorError",
    "LightnessOutOfRangeError",
]

# ---------------------------------------------------------------------------
# 错误类型（均可被 ``except ValueError`` 捕获，兼容既有测试；彼此可区分）
# ---------------------------------------------------------------------------


class GamutError(ValueError):
    """gamut 所有输入错误的基类（仍是 ValueError）。"""


class ColorFormatError(GamutError, TypeError):
    """颜色不是 3 个可转成实数的分量（长度错误、类型错误等）。"""


class NonFiniteColorError(GamutError):
    """颜色分量里有 NaN 或 Inf。"""


class LightnessOutOfRangeError(GamutError):
    """亮度（OKLab L）越界，无法在保持亮度与色相的前提下映射进 sRGB。"""


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_EPS = 1e-12          # 色域判定容差（仅用于比较浮点边界）
_CHROMA_EPS = 1e-9    # 视为中性色的色度上限（用于重建）
_NEUTRAL_EPS = 1e-6   # 亮度越界判定时视为"中性色"的色度上限
_CACHE_SIZE = 8192

# 线性 sRGB 三行系数：线性 RGB = ROWS[i] · (l, m, s)
_ROWS = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)

_cbrt = math.cbrt
_hypot = math.hypot
_sqrt = math.sqrt
_isfinite = math.isfinite

# 解码时的数值上限：远大于任何合理 sRGB 输入，只用来防止 |c|^2.4 之类的
# 幂运算在极端有限输入（如 1e308）上溢出成 inf/NaN。
_LINEAR_CAP = 1.0e300


def _pow24(v):
    """v ** 2.4；溢出时饱和到 _LINEAR_CAP。"""
    try:
        return v ** 2.4
    except OverflowError:
        return _LINEAR_CAP


def _srgb_to_linear(c):
    """sRGB 编码值 -> 线性值（带符号扩展：先取 |c| 再还原符号）。

    极端有限输入（例如 1e200）的幂运算会溢出，这里饱和到合理的数值上限，
    保证解码对任意有限输入都有界。
    """
    if c > 0.04045:
        v = _pow24((c + 0.055) / 1.055)
        return v if v < _LINEAR_CAP else _LINEAR_CAP
    if c < -0.04045:
        v = _pow24((-c + 0.055) / 1.055)
        return -v if v < _LINEAR_CAP else -_LINEAR_CAP
    return c / 12.92


def _linear_to_srgb(c):
    """线性值 -> sRGB 编码值（带符号扩展）。"""
    if -0.0031308 <= c <= 0.0031308:
        return c * 12.92
    if c > 0.0:
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return -(1.055 * ((-c) ** (1.0 / 2.4)) - 0.055)


def _linear_to_oklab(r, g, b):
    """线性 sRGB -> OKLab (L, a, b)。"""
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = _cbrt(l)
    m_ = _cbrt(m)
    s_ = _cbrt(s)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _oklab_to_linear(L, a, b):
    """OKLab -> 线性 sRGB，分别返回 r、g、b。"""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    r0, r1, r2 = _ROWS[0]
    g0, g1, g2 = _ROWS[1]
    b0, b1, b2 = _ROWS[2]
    return (
        r0 * l + r1 * m + r2 * s,
        g0 * l + g1 * m + g2 * s,
        b0 * l + b1 * m + b2 * s,
    )


def _cubic_roots(c3, c2, c1, c0):
    """实系数三次方程的全部实根（Cardano）。

    ``c3 == 0`` 时退化为二次/一次。返回插值用的根元组；调用方再用一步
    Newton 修正提升精度。
    """
    if c3 == 0.0:
        if c2 == 0.0:
            if c1 == 0.0:
                return ()
            return (-c0 / c1,)
        disc = c1 * c1 - 4.0 * c2 * c0
        if disc < 0.0:
            return ()
        sq = _sqrt(disc)
        # 数值稳定的二次求根（避免相近根相消）。
        q = -0.5 * (c1 + math.copysign(sq, c1))
        if q == 0.0:
            return (0.0,)
        return (q / c2, c0 / q)
    return _cardano(c3, c2, c1, c0)


def _cardano(c3, c2, c1, c0):
    inv = 1.0 / c3
    b = c2 * inv
    c = c1 * inv
    third = b / 3.0
    p3 = (c - b * third) / 3.0
    p3cubed = p3 * p3 * p3
    q = 2.0 * third * third * third - third * c + c0 * inv
    half = 0.5 * q
    disc = half * half + p3cubed
    if disc > 0.0:
        sd = _sqrt(disc)
        return (_cbrt(-half + sd) + _cbrt(-half - sd) - third,)
    if disc == 0.0:
        u = _cbrt(-half)
        return (2.0 * u - third, -u - third)
    r = _sqrt(-p3cubed)
    phi = math.acos(max(-1.0, min(1.0, -half / r)))
    m = 2.0 * _sqrt(-p3)
    t = math.pi * 2.0 / 3.0
    phi3 = phi / 3.0
    return (
        m * math.cos(phi3) - third,
        m * math.cos(phi3 - t) - third,
        m * math.cos(phi3 + t) - third,
    )


def _channel_polys(L, ca, sa):
    """固定 (L, 色相方向) 时三个线性通道对色度 C 的三次多项式系数。

    返回三元组 ``(p_r, p_g, p_b)``，每个是 ``(c0, c1, c2, c3)``，
    满足 channel(C) = c0 + c1*C + c2*C² + c3*C³。
    """
    a0 = 0.3963377774 * ca + 0.2158037573 * sa
    a1 = -0.1055613458 * ca - 0.0638541728 * sa
    a2 = -0.0894841775 * ca - 1.2914855480 * sa
    l2 = L * L
    l3 = l2 * L
    out = []
    for p, q, r in _ROWS:
        out.append((
            (p + q + r) * l3,
            3.0 * l2 * (p * a0 + q * a1 + r * a2),
            3.0 * L * (p * a0 * a0 + q * a1 * a1 + r * a2 * a2),
            p * a0 * a0 * a0 + q * a1 * a1 * a1 + r * a2 * a2 * a2,
        ))
    return out


def _poly_eval(co, x):
    return ((co[3] * x + co[2]) * x + co[1]) * x + co[0]


def _poly_deriv(co, x):
    return (3.0 * co[3] * x + 2.0 * co[2]) * x + co[1]


def _solve_bound(c0, c1, c2, c3, inv, third, p3, p3cubed, base, two_pi, bound, C):
    """求 c0+c1*C+c2*C²+c3*C³ = bound 中落在 (0, C) 内的根（元组）。"""
    half = 0.5 * (base + (c0 - bound) * inv)
    disc = half * half + p3cubed
    if disc > 0.0:
        sd = _sqrt(disc)
        candidates = (_cbrt(-half + sd) + _cbrt(-half - sd) - third,)
    elif disc == 0.0:
        u = _cbrt(-half)
        candidates = (2.0 * u - third, -u - third)
    else:
        r = _sqrt(-p3cubed)
        phi = math.acos(max(-1.0, min(1.0, -half / r)))
        m = 2.0 * _sqrt(-p3)
        # 三分角公式：C = m*cos((phi+2πk)/3) - b/3。
        candidates = (
            m * math.cos(phi / 3.0) - third,
            m * math.cos((phi - two_pi) / 3.0) - third,
            m * math.cos((phi + two_pi) / 3.0) - third,
        )
    return tuple(x for x in candidates if 0.0 < x < C)


def _is_monotone(co, C):
    """通道多项式在 [0, C] 上是否单调（导数不变号）。"""
    # d(x) = 3 c3 x² + 2 c2 x + c1
    c1, c2, c3 = co[1], co[2], co[3]
    a = 3.0 * c3
    b = 2.0 * c2
    d0 = c1
    dC = a * C * C + b * C + c1
    if (d0 > 0.0) == (dC > 0.0):
        # 端点同号；再看顶点是否落在区间内且反号。
        if a != 0.0:
            xv = -b / (2.0 * a)
            if 0.0 < xv < C:
                dv = (a * xv + b) * xv + c1
                if (dv > 0.0) != (d0 > 0.0):
                    return False
        return True
    return False


def _channel_exits(co, C):
    """通道在 [0, C] 内离开 [0, 1] 的边界位置；从不离开返回 None。

    在 C=0 处通道值等于 L³（L ∈ [0, 1] 时在区间内）。返回
    ``(exit, bound)``：exit 是越界段左端点，bound 是该段右端交点对应的
    边界值（0 或 1），供 Newton 修正使用。

    单调通道只可能从一侧越界，用带括号保护的 Newton 直接求根；非单调
    通道把 (0, C) 按所有交点切成若干段，逐段检查（Cardano 闭式根）。
    """
    c0, c1, c2, c3 = co
    end = ((c3 * C + c2) * C + c1) * C + c0

    if _is_monotone(co, C):
        if end > 1.0 + _EPS:
            bound = 1.0
        elif end < -_EPS:
            bound = 0.0
        else:
            return None  # f(0)=L³∈[0,1] 且单调、C 处也在区间内：全程在区间内
        x = _monotone_root(c0, c1, c2, c3, C, bound)
        if x is None:
            return 0.0, bound
        return x, bound

    if c3 == 0.0:
        return _channel_exits_deg(co, C)
    inv = 1.0 / c3
    b = c2 * inv
    c = c1 * inv
    third = b / 3.0
    p3 = (c - b * third) / 3.0
    p3cubed = p3 * p3 * p3
    base = 2.0 * third * third * third - third * c

    two_pi = math.pi * 2.0
    roots = []
    for bound in (0.0, 1.0):
        for x in _solve_bound(c0, c1, c2, c3, inv, third, p3, p3cubed,
                              base, two_pi, bound, C):
            roots.append((x, bound))
    if not roots:
        # 没有落在 (0, C) 内的交点；仍要检查是否有不在 (0, C) 的根使
        # 通道在 C 处越界（例如唯一的交点恰好在 C 外侧但 C 已越界）。
        if end < -_EPS or end > 1.0 + _EPS:
            return 0.0, (0.0 if end < 0.0 else 1.0)
        return None
    roots.sort()
    prev = 0.0
    for x, bound in roots:
        mid = 0.5 * (prev + x)
        if mid > 0.0:
            v = ((c3 * mid + c2) * mid + c1) * mid + c0
            if v < -_EPS or v > 1.0 + _EPS:
                return prev, bound
        prev = x
    mid = 0.5 * (prev + C)
    if mid > 0.0:
        v = ((c3 * mid + c2) * mid + c1) * mid + c0
        if v < -_EPS or v > 1.0 + _EPS:
            return prev, roots[-1][1]
    return None


def _monotone_root(c0, c1, c2, c3, C, bound):
    """单调通道在 (0, C) 内 f(x)=bound 的根。

    调用方已保证 f 在 [0, C] 上单调且 f(0)=L³∈[0,1]、f(C) 越界，因此根
    唯一且落在开区间内。用带括号保护的 Newton，括号用 f 的单调方向维护。
    """
    f0 = c0 - bound
    if f0 == 0.0:
        return 0.0  # f(0) 恰在边界上（例如 L=1 且 bound=1），根在 0
    increasing = f0 < 0.0  # 越界一侧是 f(C) > 1（bound=1）或 f(C) < 0（bound=0）
    lo = 0.0
    hi = C
    x = 0.5 * C
    for _ in range(40):
        fx = ((c3 * x + c2) * x + c1) * x + c0 - bound
        if fx == 0.0:
            return x
        if (fx > 0.0) == increasing:
            hi = x
        else:
            lo = x
        dfx = (3.0 * c3 * x + 2.0 * c2) * x + c1
        nx = x - fx / dfx if dfx != 0.0 else 0.5 * (lo + hi)
        if not (lo < nx < hi):
            nx = 0.5 * (lo + hi)
        if nx == x:
            return nx
        x = nx
    return x


def _channel_exits_deg(co, C):
    """退化（三次项为 0）时的边界求取。"""
    roots = []
    for bound in (0.0, 1.0):
        for root in _cubic_roots(co[3], co[2], co[1], co[0] - bound):
            if 0.0 < root < C:
                roots.append((root, bound))
    if not roots:
        v = _poly_eval(co, C)
        if v < -_EPS or v > 1.0 + _EPS:
            return 0.0, (0.0 if v < 0.0 else 1.0)
        return None
    roots.sort()
    prev = 0.0
    prev_bound = roots[0][1]
    for x, bound in roots:
        mid = 0.5 * (prev + x)
        if mid > 0.0:
            v = _poly_eval(co, mid)
            if v < -_EPS or v > 1.0 + _EPS:
                return prev, prev_bound
        prev = x
        prev_bound = bound
    mid = 0.5 * (prev + C)
    if mid > 0.0:
        v = _poly_eval(co, mid)
        if v < -_EPS or v > 1.0 + _EPS:
            return prev, prev_bound
    return None


def _refine_exit(co, x, bound):
    """把边界点 x 微调到通道恰好触到 bound 的位置（一步 Newton）。"""
    f = _poly_eval(co, x) - bound
    d = _poly_deriv(co, x)
    if d != 0.0:
        nx = x - f / d
        if 0.0 <= nx < x:
            return nx
    return x


def _max_chroma(L, ca, sa, C):
    """固定 L 与色相下 sRGB 允许的最大色度（不超过 C）。"""
    polys = _channel_polys(L, ca, sa)
    best = C
    best_poly = None
    best_bound = 0.0
    for co in polys:
        hit = _channel_exits(co, C)
        if hit is not None and hit[0] < best:
            best, best_bound = hit
            best_poly = co
    if best_poly is not None and best > 0.0:
        best = _refine_exit(best_poly, best, best_bound)
    if best < 0.0:
        return 0.0
    return best


def _map_validated(r, g, b):
    """映射一个已校验（有限）sRGB 三元组，返回 tuple[float, float, float]。"""
    # 1) 已在色域内：原样返回（逐位相同）。
    if 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0:
        return (r, g, b)

    # 2) 转 OKLab，取出亮度 L、色度 C 与单位色相方向 (ca, sa)。
    L, a, bb = _linear_to_oklab(
        _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    )
    C = _hypot(a, bb)

    if L < 0.0:
        raise LightnessOutOfRangeError(
            "OKLab 亮度 %r < 0，无法在保持亮度与色相的前提下映射" % (L,)
        )
    if L > 1.0:
        if C <= _NEUTRAL_EPS:
            raise LightnessOutOfRangeError(
                "中性色的 OKLab 亮度 %r 超出 [0, 1]，无法在保持亮度"
                "与色相的前提下映射" % (L,)
            )
        # 彩色且亮度 > 1：sRGB 不含该亮度，压到 1 后按色相收缩色度。
        L = 1.0

    if C <= _CHROMA_EPS:
        # [0, 1] 内的中性色（只可能来自舍入残差）：三通道相等。
        v = _linear_to_srgb(L * L * L)
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        return (v, v, v)

    ca = a / C
    sa = bb / C
    cm = _max_chroma(L, ca, sa, C)

    # 3) 用收缩后的色度重建线性 sRGB，并做边界钳制。
    lr, lg, lb = _oklab_to_linear(L, cm * ca, cm * sa)
    if lr < 0.0:
        lr = 0.0
    elif lr > 1.0:
        lr = 1.0
    if lg < 0.0:
        lg = 0.0
    elif lg > 1.0:
        lg = 1.0
    if lb < 0.0:
        lb = 0.0
    elif lb > 1.0:
        lb = 1.0
    return (_linear_to_srgb(lr), _linear_to_srgb(lg), _linear_to_srgb(lb))


_map_validated = functools.lru_cache(maxsize=_CACHE_SIZE)(_map_validated)


def _as_real(value, index):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ColorFormatError(
            "颜色的第 %d 个分量不是实数: %r" % (index, value)
        ) from None


def _parse_color(color):
    """校验颜色输入，返回 (r, g, b) 三元 float；错误分类抛出。"""
    if type(color) is tuple:
        if len(color) != 3:
            raise ColorFormatError("颜色必须是恰好 3 个数值分量")
        r, g, b = color
        # 快路径：三个分量都是 float 时直接检查有限性。
        if type(r) is float and type(g) is float and type(b) is float:
            if _isfinite(r) and _isfinite(g) and _isfinite(b):
                return r, g, b
            raise NonFiniteColorError("颜色分量必须是有限值（不允许 NaN/Inf）")
    elif isinstance(color, (str, bytes, bytearray)):
        raise ColorFormatError("颜色必须是 3 个数值分量，不能是字符串")
    try:
        r, g, b = color
    except (TypeError, ValueError):
        raise ColorFormatError("颜色必须是恰好 3 个数值分量") from None
    r = _as_real(r, 0)
    g = _as_real(g, 1)
    b = _as_real(b, 2)
    if not (_isfinite(r) and _isfinite(g) and _isfinite(b)):
        raise NonFiniteColorError("颜色分量必须是有限值（不允许 NaN/Inf）")
    return r, g, b


class GamutMap:
    """sRGB 色域映射器（无可变状态，可被多线程共享）。"""

    __slots__ = ()

    def map(self, color):
        """把 sRGB 颜色 ``(r, g, b)`` 映射到 sRGB 色域。

        - 已在色域内：原样返回 ``(r, g, b)``；
        - 越界：按 OKLCh 收缩色度，保持亮度与色相；
        - 非有限输入抛 :class:`NonFiniteColorError`；
        - 长度/类型错误抛 :class:`ColorFormatError`；
        - 亮度无法保持时抛 :class:`LightnessOutOfRangeError`。

        返回 ``tuple[float, float, float]``，各分量在 [0, 1]。
        """
        r, g, b = _parse_color(color)
        return _map_validated(r, g, b)
