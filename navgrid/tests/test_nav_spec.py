"""NavGrid 约束测试：与既有 tests/test_core.py 并列，不改动任何既有测试。

覆盖题面的隐藏要点：
- 路径总代价与独立实现的 Dijkstra 一致（按地形加权，不是步数），
  对角步长为 ``sqrt(2)``，对角不贴角穿墙；
- 平滑用超覆盖视线拉直：结果仍无碰撞、点数不增、代价不增；
- 不可达返回 ``None``；
- 非法网格 / 负代价 / 非有限代价给出可区分的异常类型；
- 并发调用同一实例安全；
- 1024 格规模寻路加平滑远低于 300 ms。
"""

import heapq
import math
import random
import threading
import time

import pytest

import navgrid
from navgrid import (
    InvalidGridError,
    NegativeCostError,
    NavGrid,
    NonFiniteCostError,
)

SQRT2 = math.sqrt(2.0)
FREE = 0
WALL = 1


# ---------------------------------------------------------------------------
# 独立参考实现：逐点 Dijkstra（与 A* 代码路径完全不同）
# ---------------------------------------------------------------------------

def _reference_cost(grid, start, goal):
    """带地形倍率的 8 邻域 Dijkstra，返回最优代价；不可达返回 None。"""
    h = len(grid)
    w = len(grid[0])
    inf = math.inf

    def passable(x, y):
        return 0 <= x < w and 0 <= y < h and grid[y][x] != WALL

    if not passable(*start) or not passable(*goal):
        return None

    dist = {(start[0], start[1]): 0.0}
    pq = [(0.0, start[0], start[1])]
    while pq:
        d, x, y = heapq.heappop(pq)
        if d > dist.get((x, y), inf) + 1e-12:
            continue
        if (x, y) == (goal[0], goal[1]):
            return d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if not passable(nx, ny):
                continue
            if dx and dy and not (passable(x + dx, y) and passable(x, y + dy)):
                continue  # 不贴角穿墙
            step = SQRT2 if (dx and dy) else 1.0
            cell = grid[ny][nx]
            mult = 1.0 if cell == FREE else cell
            nd = d + step * mult
            if nd < dist.get((nx, ny), inf) - 1e-12:
                dist[(nx, ny)] = nd
                heapq.heappush(pq, (nd, nx, ny))
    return None


def _path_cost(grid, path):
    """按“步长 × 进入格倍率”累计路径代价。"""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        cell = grid[y1][x1]
        mult = 1.0 if cell == FREE else cell
        total += math.hypot(x1 - x0, y1 - y0) * mult
    return total


def _supercover_cells(x0, y0, x1, y1):
    """独立实现的超覆盖判定：线段擦过的所有格。

    直接按参数采样格线穿越点，与被测实现不是同一份代码。
    """
    dx = x1 - x0
    dy = y1 - y0
    nx = abs(dx)
    ny = abs(dy)
    cells = {(x0, y0)}
    steps = max(nx, ny) * 2 + 1
    for k in range(1, steps + 1):
        t = k / steps
        fx = x0 + dx * t
        fy = y0 + dy * t
        for ix in (math.floor(fx + 0.5), math.ceil(fx - 0.5)):
            for iy in (math.floor(fy + 0.5), math.ceil(fy - 0.5)):
                cells.add((ix, iy))
    return cells


def _random_grid(rnd, w, h, wall_p=0.2, weight_p=0.3, weight_max=3.5):
    g = []
    for _ in range(h):
        row = []
        for _ in range(w):
            q = rnd.random()
            if q < wall_p:
                row.append(WALL)
            elif q < wall_p + (1.0 - wall_p) * (1.0 - weight_p):
                row.append(FREE)
            else:
                row.append(round(rnd.uniform(0.3, weight_max), 3))
        g.append(row)
    return g


def _free_cells(grid):
    return [(x, y) for y in range(len(grid)) for x in range(len(grid[0]))
            if grid[y][x] != WALL]


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

GRIDS = []
for _seed in range(1, 13):
    _rnd = random.Random(_seed)
    GRIDS.append(_random_grid(_rnd, _rnd.randint(4, 16), _rnd.randint(4, 16)))


@pytest.mark.parametrize("seed", range(len(GRIDS)))
def test_nav_cost_matches_dijkstra(seed):
    """路径代价与 Dijkstra 最优代价一致（加权，不是步数）。"""
    grid = GRIDS[seed]
    rnd = random.Random(1000 + seed)
    free = _free_cells(grid)
    ng = NavGrid(grid)
    for _ in range(15):
        start = rnd.choice(free)
        goal = rnd.choice(free)
        got = ng.astar(start, goal)
        want = _reference_cost(grid, start, goal)
        assert (got is None) == (want is None)
        if got is not None:
            assert abs(_path_cost(grid, got) - want) <= 1e-9


def test_nav_diagonal_cost_is_sqrt2_and_no_corner_cutting():
    """空地斜线走对角，代价是 sqrt(2) 的整数倍；正方墙夹角不可斜穿。"""
    open_grid = [[FREE] * 6 for _ in range(6)]
    path = NavGrid(open_grid).astar((0, 0), (5, 5))
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert abs(_path_cost(open_grid, path) - 5 * SQRT2) <= 1e-9

    # 两堵墙在 (2,1) / (1,2) 形成夹角：(1,1) -> (2,2) 不能一步斜穿，
    # 必须绕行，路径代价因而大于 sqrt(2)。
    corner = [
        [FREE, FREE, FREE, FREE],
        [FREE, FREE, WALL, FREE],
        [FREE, WALL, FREE, FREE],
        [FREE, FREE, FREE, FREE],
    ]
    ng = NavGrid(corner)
    path = ng.astar((1, 1), (2, 2))
    assert path is not None  # 绕行可达
    assert abs(_path_cost(corner, path) - _reference_cost(corner, (1, 1), (2, 2))) <= 1e-9
    assert _path_cost(corner, path) > SQRT2


def test_nav_unreachable_returns_none():
    grid = [[FREE, WALL, FREE]]
    assert NavGrid(grid).astar((0, 0), (2, 0)) is None
    assert NavGrid([[WALL]]).astar((0, 0), (0, 0)) is None


def test_nav_start_equals_goal():
    assert NavGrid([[FREE]]).astar((0, 0), (0, 0)) == [(0, 0)]


def test_nav_smoothing_is_valid_and_not_costlier():
    """平滑结果无碰撞、点数不增、代价不增。"""
    for seed, grid in enumerate(GRIDS):
        rnd = random.Random(2000 + seed)
        free = _free_cells(grid)
        ng = NavGrid(grid)
        for _ in range(6):
            start = rnd.choice(free)
            goal = rnd.choice(free)
            raw = ng.astar(start, goal)
            if raw is None or len(raw) < 2:
                continue
            smooth = ng.smooth(raw)
            assert smooth[0] == raw[0] and smooth[-1] == raw[-1]
            assert len(smooth) <= len(raw)
            assert _path_cost(grid, smooth) <= _path_cost(grid, raw) + 1e-9
            h, w = len(grid), len(grid[0])
            for a, b in zip(smooth, smooth[1:]):
                assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 1
                for x, y in _supercover_cells(a[0], a[1], b[0], b[1]):
                    if not (0 <= x < w and 0 <= y < h):
                        continue
                    assert grid[y][x] != WALL, (seed, a, b, (x, y))


def test_nav_smooth_straight_line_collapses():
    grid = [[FREE] * 10 for _ in range(10)]
    ng = NavGrid(grid)
    path = ng.astar((0, 0), (9, 9))
    smooth = ng.smooth(path)
    assert smooth == [(0, 0), (9, 9)]


def test_nav_smooth_empty_and_single():
    ng = NavGrid([[FREE, FREE]])
    assert ng.smooth([]) == []
    assert ng.smooth([(1, 0)]) == [(1, 0)]


def test_nav_smooth_rejects_colliding_input():
    ng = NavGrid([[FREE, WALL, FREE]])
    with pytest.raises(ValueError):
        ng.smooth([(0, 0), (1, 0)])


def test_nav_invalid_grid_errors_distinguishable():
    bad_grids = [
        [],                            # 空
        [[FREE, FREE], [FREE]],        # 异形
        [["x", FREE]],                 # 非数值
        [[FREE, None]],
    ]
    for g in bad_grids:
        with pytest.raises(InvalidGridError):
            NavGrid(g)


def test_nav_negative_cost_error_distinguishable():
    with pytest.raises(NegativeCostError):
        NavGrid([[FREE, -1]])
    # 三类错误类型互不相同，可按类型区分。
    assert not issubclass(NegativeCostError, (InvalidGridError, NonFiniteCostError))


def test_nav_non_finite_cost_error_distinguishable():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(NonFiniteCostError):
            NavGrid([[FREE, bad]])
    assert not issubclass(NonFiniteCostError, (InvalidGridError, NegativeCostError))


def test_nav_all_errors_are_value_errors():
    for exc in (InvalidGridError, NegativeCostError, NonFiniteCostError):
        assert issubclass(exc, ValueError)


def test_nav_bad_coordinates_raise():
    ng = NavGrid([[FREE, FREE]])
    with pytest.raises(ValueError):
        ng.astar((0, 0), (9, 0))
    with pytest.raises(ValueError):
        ng.astar((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        ng.astar((0.0, 0.0), (1, 0))


def test_nav_deterministic():
    grid = _random_grid(random.Random(77), 20, 20)
    ng = NavGrid(grid)
    free = _free_cells(grid)
    rnd = random.Random(78)
    for _ in range(10):
        s = rnd.choice(free)
        t = rnd.choice(free)
        first = ng.astar(s, t)
        assert ng.astar(s, t) == first
        if first is not None:
            assert ng.smooth(first) == ng.smooth(first)


def test_nav_thread_safety():
    grid = _random_grid(random.Random(5), 40, 40)
    grid[0][0] = FREE
    ng = NavGrid(grid)
    free = _free_cells(grid)
    rnd = random.Random(6)
    queries = [(rnd.choice(free), rnd.choice(free)) for _ in range(60)]
    expected = [(ng.astar(s, t), ng.smooth(ng.astar(s, t)) if ng.astar(s, t) else None)
                for s, t in queries]

    results = [None] * len(queries)
    errors = []

    def worker(slots):
        try:
            for k in slots:
                s, t = queries[k]
                p = ng.astar(s, t)
                results[k] = (p, ng.smooth(p) if p else None)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(range(w, len(queries), 4),))
               for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results == expected


def test_nav_performance_1024_cells():
    """32x32（1024 格）寻路 + 平滑远低于 300 ms。"""
    grid = _random_grid(random.Random(3), 32, 32, wall_p=0.2, weight_p=0.3)
    grid[0][0] = FREE
    grid[31][31] = FREE
    ng = NavGrid(grid)
    start = time.perf_counter()
    for _ in range(20):
        path = ng.astar((0, 0), (31, 31))
        if path is not None:
            ng.smooth(path)
    elapsed = time.perf_counter() - start
    assert elapsed / 20 < 0.3
