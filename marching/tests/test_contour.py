"""marching.contour 的真实行为契约测试。"""

import math
import random
import time

import pytest

import marching


def _contour(field, level=0.5):
    return marching.Marching().contour(field, level)


def _assert_closed_loops(out, w, h):
    for loop in out:
        assert len(loop) >= 4
        assert loop[0] == loop[-1]
        for x, y in loop:
            assert math.isfinite(x) and math.isfinite(y)
            assert -1e-8 <= x <= w - 1 + 1e-8
            assert -1e-8 <= y <= h - 1 + 1e-8
        for (x0, y0), (x1, y1) in zip(loop, loop[1:]):
            # 每段最多跨越一个单元的对角线
            assert math.hypot(x1 - x0, y1 - y0) <= math.sqrt(2) + 1e-6


# ---------------------------------------------------------------- 闭合性

def test_returns_closed_polyline():
    out = _contour([[0.0, 1.0], [1.0, 0.0]])
    assert out
    _assert_closed_loops(out, 2, 2)


def test_interior_ring_is_closed():
    field = [[0.0] * 5 for _ in range(5)]
    for y in range(1, 4):
        for x in range(1, 4):
            field[y][x] = 1.0
    out = _contour(field)
    assert len(out) == 1
    _assert_closed_loops(out, 5, 5)
    xs = [p[0] for p in out[0]]
    ys = [p[1] for p in out[0]]
    assert min(xs) == 0.5 and max(xs) == 3.5
    assert min(ys) == 0.5 and max(ys) == 3.5


def test_open_contour_closes_along_boundary():
    out = _contour([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    assert len(out) == 1
    _assert_closed_loops(out, 2, 3)


# ---------------------------------------------------------------- 鞍点

def _segments(loop):
    return list(zip(loop, loop[1:]))


def _crosses_cell_center(loop, cx, cy):
    """线段在单元内部是否穿过中心附近（实心区）。"""
    for (x0, y0), (x1, y1) in _segments(loop):
        # 是否与单元内部的中心小方块相交
        lo_x, hi_x = cx - 0.25, cx + 0.25
        lo_y, hi_y = cy - 0.25, cy + 0.25
        t0, t1 = 0.0, 1.0
        ok = True
        for d, o, lo, hi in ((x1 - x0, x0, lo_x, hi_x),
                             (y1 - y0, y0, lo_y, hi_y)):
            if abs(d) < 1e-15:
                if o <= lo or o >= hi:
                    ok = False
                    break
            else:
                a, b = (lo - o) / d, (hi - o) / d
                if a > b:
                    a, b = b, a
                t0, t1 = max(t0, a), min(t1, b)
                if t0 >= t1:
                    ok = False
                    break
        if ok:
            return True
    return False


@pytest.mark.parametrize("field,center_high", [
    ([[0.0, 1.0], [1.0, 0.0]], True),
    ([[1.0, 0.0], [0.0, 1.0]], True),
    ([[0.4, 1.0], [1.0, 0.4]], True),
    ([[1.0, 0.4], [0.4, 1.0]], True),
    ([[0.0, 0.6], [0.6, 0.0]], False),
    ([[0.6, 0.0], [0.0, 0.6]], False),
])
def test_saddle_never_crosses_solid(field, center_high):
    out = _contour(field)
    _assert_closed_loops(out, 2, 2)
    # 单元中心 (0.5, 0.5)。中心实心时环包围中心，等值线不得穿过中心。
    assert not _crosses_cell_center(out[0], 0.5, 0.5)
    # 中心实心 => 一个环；中心空 => 两个独立小环
    assert (len(out) == 1) == center_high


def test_saddle_exact_center_treated_as_solid():
    # 中心值恰为 level：按实心处理，等值线仍不穿越实心角
    out = _contour([[0.0, 1.0], [1.0, 0.0]])
    assert len(out) == 1
    assert not _crosses_cell_center(out[0], 0.5, 0.5)


def test_checkerboard_has_no_solid_crossings():
    field = [[float((x + y) % 2) for x in range(9)] for y in range(9)]
    out = _contour(field)
    _assert_closed_loops(out, 9, 9)
    for loop in out:
        for (x0, y0), (x1, y1) in _segments(loop):
            # 任何线段的采样中点都不能落在实心角点邻域
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                for cy in range(9):
                    for cx in range(9):
                        if field[cy][cx] >= 0.5:
                            assert not (abs(x - cx) < 1e-6 and abs(y - cy) < 1e-6)


# ---------------------------------------------------------------- 空场与退化

@pytest.mark.parametrize("field", [
    [],
    [[]],
    [[0.0]],
    [[0.0, 1.0]],
    [[0.0, 0.0], [0.0, 0.0]],
    [[1.0, 1.0], [1.0, 1.0]],
    [[0.5, 0.5], [0.5, 0.5]],
])
def test_degenerate_fields_return_empty(field):
    assert _contour(field) == []


@pytest.mark.parametrize("level", [-10.0, 2.0])
def test_level_outside_range_returns_empty(level):
    assert _contour([[0.0, 1.0], [1.0, 0.0]], level) == []


# ---------------------------------------------------------------- 边界单元

def test_boundary_cell_points_within_field():
    field = [[0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]]
    out = _contour(field)
    assert out
    _assert_closed_loops(out, 3, 4)


def test_contour_clipped_at_edge_is_closed():
    field = [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    out = _contour(field)
    assert out
    _assert_closed_loops(out, 2, 3)


# ---------------------------------------------------------------- 非法输入

@pytest.mark.parametrize("field", [
    None,
    "abc",
    5,
    [1.0, 2.0],
    [[1.0, "a"]],
    [[1.0, None]],
    [[1.0, float("nan")]],
    [[1.0, float("inf")]],
    [[1.0, 2.0], [3.0]],
    [[1.0, [2.0]]],
])
def test_invalid_field_raises(field):
    with pytest.raises(ValueError):
        _contour(field)


@pytest.mark.parametrize("level", [
    float("nan"), float("inf"), float("-inf"), None, "x", [1.0],
])
def test_invalid_level_raises(level):
    with pytest.raises(ValueError):
        _contour([[0.0, 1.0], [1.0, 0.0]], level)


# ---------------------------------------------------------------- 其它契约

def test_returns_list_of_point_lists():
    out = _contour([[0.0, 1.0], [1.0, 0.0]])
    assert isinstance(out, list)
    assert all(isinstance(loop, list) for loop in out)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in out[0])


def test_deterministic():
    field = [[random.Random(7).random() for _ in range(6)] for _ in range(6)]
    assert _contour(field) == _contour(field)


def test_input_not_mutated():
    field = [[0.0, 1.0], [1.0, 0.0]]
    snapshot = [row[:] for row in field]
    _contour(field)
    assert field == snapshot


def test_concurrent_calls_agree():
    from concurrent.futures import ThreadPoolExecutor
    field = [[0.0, 1.0], [1.0, 0.0]]
    expected = _contour(field)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _contour(field), range(100)))
    assert all(r == expected for r in results)


def test_10k_cells_under_400ms():
    rng = random.Random(0)
    field = [[rng.random() for _ in range(101)] for _ in range(101)]
    start = time.perf_counter()
    out = _contour(field)
    elapsed = time.perf_counter() - start
    assert out
    assert elapsed < 0.4, "%.1f ms" % (elapsed * 1000.0)


def test_10k_cells_checkerboard_under_400ms():
    n = 101
    field = [[float((x + y) % 2) for x in range(n)] for y in range(n)]
    start = time.perf_counter()
    out = _contour(field)
    elapsed = time.perf_counter() - start
    assert out
    assert elapsed < 0.4, "%.1f ms" % (elapsed * 1000.0)


@pytest.mark.parametrize("seed", range(5))
def test_fuzz_degenerate_fields_stay_closed(seed):
    """含大量恰等于 level 的顶点时，输出仍必须是合法闭合折线。"""
    rng = random.Random(seed)
    levels = [0.0, 0.5, 1.0]
    for _ in range(300):
        h = rng.randint(1, 4)
        w = rng.randint(1, 4)
        field = [[rng.choice(levels) for _ in range(w)] for _ in range(h)]
        _assert_closed_loops(_contour(field), w, h)
