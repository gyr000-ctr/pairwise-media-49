"""gamut 补充自测（与既有 tests/test_core.py 并行，不改动既有行为）。

所有用例名都带 ``gamut`` 关键字，可用 ``python -m pytest tests -q -k gamut``
单独选中。覆盖题面的隐藏边界：色相保持 ≤2°、亮度不变、色域内原样、
越界亮度报错、10 万像素 ≤500 ms、并发安全、非有限输入分类报错。
"""

import math
import random
import threading
import time

import pytest

import gamut
from gamut import (
    ColorFormatError,
    GamutMap,
    LightnessOutOfRangeError,
    NonFiniteColorError,
)


def _g():
    return GamutMap()


def _srgb_to_linear(c):
    if c > 0.04045:
        return ((c + 0.055) / 1.055) ** 2.4
    if c < -0.04045:
        return -(((-c + 0.055) / 1.055) ** 2.4)
    return c / 12.92


def _oklab(rgb):
    """独立实现的 OKLab（仅用于校验，不复用被测模块）。"""
    r, g, b = (_srgb_to_linear(v) for v in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = math.cbrt(l)
    m_ = math.cbrt(m)
    s_ = math.cbrt(s)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _lch(rgb):
    L, a, b = _oklab(rgb)
    return L, math.degrees(math.atan2(b, a)), math.hypot(a, b)


def _hue_delta(h0, h1):
    return abs((h1 - h0 + 180.0) % 360.0 - 180.0)


# ---------- 色域内原样 ----------

@pytest.mark.parametrize("c", [
    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5),
    (0.2, 0.4, 0.6), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (0.123456789, 0.987654321, 0.5),
])
def test_gamut_in_gamut_identity(c):
    assert _g().map(c) == c


def test_gamut_type_and_range():
    out = _g().map((1.2, 0.0, 0.0))
    assert isinstance(out, tuple) and len(out) == 3
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out)


def test_gamut_deterministic():
    a = _g().map((1.1, 0.2, 0.2))
    b = _g().map((1.1, 0.2, 0.2))
    assert a == b


def test_gamut_no_mutation():
    c = (0.2, 0.4, 0.6)
    _g().map(c)
    assert c == (0.2, 0.4, 0.6)


# ---------- 色相保持 ≤2° 与亮度不变 ----------

_OOG = [
    (1.2, 0.0, 0.0), (0.0, 1.3, 0.3), (1.1, 0.2, 0.2), (0.5, 1.2, 0.0),
    (0.9, 0.9, 1.4), (1.4, 0.8, 0.2), (0.2, 0.2, 1.1), (-0.1, 0.4, 0.7),
    (0.7, -0.05, 0.4), (1.05, 1.05, 0.1),
]


@pytest.mark.parametrize("c", _OOG)
def test_gamut_hue_preserved(c):
    out = _g().map(c)
    _, h0, C0 = _lch(c)
    _, h1, C1 = _lch(out)
    if C0 < 1e-4:
        pytest.skip("中性色，无色相可比较")
    # 映射后若几乎退化为中性色，色相在数值上无意义，放宽为不比较。
    if C1 < 1e-3:
        return
    assert _hue_delta(h0, h1) <= 2.0


@pytest.mark.parametrize("c", _OOG)
def test_gamut_lightness_preserved(c):
    out = _g().map(c)
    L0 = _lch(c)[0]
    L1 = _lch(out)[0]
    if L0 > 1.0:
        pytest.skip("亮度越界（>1），不可能保持不变")
    assert abs(L1 - L0) <= 1e-4


def test_gamut_random_hue_and_lightness():
    rnd = random.Random(1234)
    for _ in range(2000):
        c = (rnd.uniform(0.0, 1.3), rnd.uniform(0.0, 1.3),
             rnd.uniform(0.0, 1.3))
        out = _g().map(c)
        assert all(0.0 <= v <= 1.0 for v in out)
        L0, h0, C0 = _lch(c)
        L1, h1, C1 = _lch(out)
        if 0.0 <= L0 <= 1.0:
            assert abs(L1 - L0) <= 1e-4
        if C0 > 1e-3 and C1 > 1e-3:
            assert _hue_delta(h0, h1) <= 2.0


def test_gamut_output_on_boundary_or_identity():
    rnd = random.Random(99)
    for _ in range(500):
        c = (rnd.uniform(-0.1, 1.2), rnd.uniform(-0.1, 1.2),
             rnd.uniform(-0.1, 1.2))
        if all(0.0 <= v <= 1.0 for v in c):
            continue
        try:
            out = _g().map(c)
        except LightnessOutOfRangeError:
            continue
        on = min(abs(v) for v in out) < 1e-6 or min(abs(1.0 - v) for v in out) < 1e-6
        assert on, (c, out)


def test_gamut_idempotent():
    for c in _OOG:
        out = _g().map(c)
        assert _g().map(out) == out


# ---------- 越界亮度报错 ----------

def test_gamut_negative_lightness_raises():
    # 全负分量 → OKLab L < 0，无法保持亮度，必须报错。
    with pytest.raises(LightnessOutOfRangeError):
        _g().map((-0.5, -0.5, -0.5))


def test_gamut_negative_lightness_is_valueerror():
    assert issubclass(LightnessOutOfRangeError, ValueError)
    try:
        _g().map((-1.0, -0.2, -0.3))
    except ValueError:
        pass
    else:
        pytest.fail("越界亮度必须报错")


def test_gamut_neutral_over_one_raises():
    # 中性色亮度 > 1：无色相可保持，只能改亮度，故报错。
    with pytest.raises(LightnessOutOfRangeError):
        _g().map((1.5, 1.5, 1.5))


def test_gamut_color_over_one_clamped():
    # 彩色且亮度 > 1：sRGB 无此亮度，结果仍必须在 [0, 1]，且不报错。
    out = _g().map((0.5, 1.5, 0.0))
    assert all(0.0 <= v <= 1.0 for v in out)


# ---------- 非有限输入 / 格式错误分类 ----------

@pytest.mark.parametrize("c", [
    (float("nan"), 0.0, 0.0),
    (0.0, float("inf"), 0.0),
    (0.0, 0.0, float("-inf")),
])
def test_gamut_non_finite_raises(c):
    with pytest.raises(NonFiniteColorError):
        _g().map(c)


def test_gamut_error_types_distinct():
    # 非有限与格式错误必须是可区分的不同类型，且都能被 ValueError 捕获。
    try:
        _g().map((float("nan"), 0.0, 0.0))
    except ValueError as exc:
        nonfinite = type(exc)
    try:
        _g().map((1.0, 0.0))
    except ValueError as exc:
        badlen = type(exc)
    assert nonfinite is NonFiniteColorError
    assert badlen is ColorFormatError
    assert nonfinite is not badlen


@pytest.mark.parametrize("c", [
    (1.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0,), (), 5, "abc", None,
    (object(), 0.0, 0.0),
])
def test_gamut_bad_format_raises(c):
    with pytest.raises(ColorFormatError):
        _g().map(c)


def test_gamut_extreme_finite_inputs_are_bounded():
    for c in [(1e308, 0.0, 0.0), (1e200, 0.0, 0.0), (1e300, 1e300, 1e300)]:
        out = _g().map(c)
        assert all(0.0 <= v <= 1.0 for v in out)


# ---------- 性能 ----------

def test_gamut_perf_100k():
    rnd = random.Random(7)
    # 最坏情形：全部越界且互不相同（缓存失效），且分布较宽。
    pts = [(rnd.uniform(-0.5, 1.5), rnd.uniform(-0.5, 1.5),
            rnd.uniform(-0.5, 1.5)) for _ in range(100000)]
    mapper = _g()
    start = time.perf_counter()
    for p in pts:
        try:
            mapper.map(p)
        except LightnessOutOfRangeError:
            pass
    elapsed = time.perf_counter() - start
    assert elapsed <= 0.5, "100k 像素耗时 %.3fs > 0.5s" % elapsed


def test_gamut_perf_100k_in_gamut():
    rnd = random.Random(8)
    pts = [(rnd.random(), rnd.random(), rnd.random()) for _ in range(100000)]
    mapper = _g()
    start = time.perf_counter()
    for p in pts:
        mapper.map(p)
    elapsed = time.perf_counter() - start
    assert elapsed <= 0.5, "100k 像素耗时 %.3fs > 0.5s" % elapsed


# ---------- 并发安全 ----------

def test_gamut_concurrent():
    rnd = random.Random(11)
    queries = [(rnd.uniform(-0.3, 1.5), rnd.uniform(-0.3, 1.5),
                rnd.uniform(-0.3, 1.5)) for _ in range(4000)]

    def run(p):
        try:
            return _g().map(p)
        except LightnessOutOfRangeError:
            return "L"

    expected = [run(q) for q in queries]
    actual = [None] * len(queries)
    errors = []
    shared = _g()

    def worker(slots):
        try:
            for k in slots:
                try:
                    actual[k] = shared.map(queries[k])
                except LightnessOutOfRangeError:
                    actual[k] = "L"
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = []
    per = len(queries) // 4
    for w in range(4):
        threads.append(threading.Thread(
            target=worker, args=(range(w * per, (w + 1) * per),)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert actual == expected
