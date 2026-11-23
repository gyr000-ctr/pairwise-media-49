"""navgrid 核心 API：带地形代价的 8 邻域 A* 与任意角度视线平滑。

坐标约定为 ``(x, y)``：x 为列、y 为行，原点在左上角，``grid[y][x]``
是该格的地形值。

地形语义
--------
* ``0`` —— 普通地面，单位长度的代价为 1；
* ``1`` —— 墙，不可通行；
* 其它有限的非负值 ``v`` —— 地形倍率，进入该格时单位长度的代价为
  ``v``，即一步的代价 = 步长 × 进入格的倍率。

正交步长为 1，对角步长为 ``sqrt(2)``。对角移动不允许贴着墙角穿过：
只有两个相邻正交格都可通行时才允许走对角。该规则与平滑使用的超覆盖
（supercover）视线判定一致，因此 A* 给出的每段路径本身也无碰撞。

实现要点
--------
* 网格在构造时一次性校验并快照为只读数组，之后 ``astar`` / ``smooth``
  只读不写，所以同一 ``NavGrid`` 可被多个线程并发使用。
* A* 使用八方向距离作为启发式（乘上全图最小地形倍率以保持可采纳且
  一致），配合 ``(f, h, 格号)`` 的堆序稳定地打破平局；所有移动代价为正，
  故首次弹出终点时的路径代价与 Dijkstra 最优代价一致。
* 平滑是字符串拉直（string pulling）：从当前锚点出发，用整数超覆盖
  Bresenham 判定视线，取最远的无碰撞且代价不增加的点作为下一锚点。
  由于超覆盖包含线段擦过的所有格，平滑后的路径仍无碰撞。
* 结果不含任何随机量，同一输入必得同一输出。
"""

import heapq
import math

__all__ = [
    "NavGrid",
    "GridError",
    "InvalidGridError",
    "NegativeCostError",
    "NonFiniteCostError",
]

_SQRT2 = math.sqrt(2.0)
# 八方向距离 = max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)
_OCTILE_K = _SQRT2 - 1.0
_INF = math.inf
_EPS = 1e-9


class GridError(ValueError):
    """网格或地形代价非法。基类本身也是 ``ValueError``。"""


class InvalidGridError(GridError):
    """网格结构非法：空、非矩形、含非数值等。"""


class NegativeCostError(GridError):
    """地形代价为负。"""


class NonFiniteCostError(GridError):
    """地形代价为 NaN 或 ±Inf。"""


def _cell_value(value, x, y):
    """把单个格值规整成 float；按错误类别抛不同的异常。"""
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)):
        raise InvalidGridError(
            "invalid grid: cell (%d, %d) is not a number: %r" % (x, y, value)
        )
    try:
        v = float(value)
    except OverflowError:
        raise NonFiniteCostError(
            "non-finite cost: cell (%d, %d) is out of float range" % (x, y)
        ) from None
    if not math.isfinite(v):
        raise NonFiniteCostError(
            "non-finite cost: cell (%d, %d) is %r" % (x, y, value)
        )
    if v < 0.0:
        raise NegativeCostError(
            "negative cost: cell (%d, %d) is %r" % (x, y, value)
        )
    return v


def _parse_grid(grid):
    """校验并展开网格，返回 ``(w, h, flat)``。"""
    if isinstance(grid, (str, bytes, bytearray)):
        raise InvalidGridError("invalid grid: expected a 2D numeric sequence")
    try:
        rows = list(grid)
    except TypeError:
        raise InvalidGridError("invalid grid: expected a 2D numeric sequence") from None
    if not rows:
        raise InvalidGridError("invalid grid: grid is empty")

    width = None
    flat = []
    for y, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise InvalidGridError(
                "invalid grid: row %d is not a sequence of numbers" % y
            )
        try:
            cells = list(row)
        except TypeError:
            raise InvalidGridError(
                "invalid grid: row %d is not a sequence of numbers" % y
            ) from None
        if not cells:
            raise InvalidGridError("invalid grid: row %d is empty" % y)
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise InvalidGridError(
                "invalid grid: row %d has %d cells, expected %d"
                % (y, len(cells), width)
            )
        for x, value in enumerate(cells):
            flat.append(_cell_value(value, x, y))
    return width, len(rows), flat


def _supercover(x0, y0, x1, y1):
    """整数超覆盖 Bresenham：依次产出线段擦过的每一个格（含两端点）。

    线段按格中心之间的参数化处理，单元格 ``(x, y)`` 覆盖
    ``[x-0.5, x+0.5] × [y-0.5, y+0.5]``。线段恰好穿过格角时，
    两侧的格也计入，从而保证“贴角穿越”会被判为擦碰。

    生成器形式：调用方可在遇到墙时立即停止，不必走完整条线段。
    """
    dx = x1 - x0
    dy = y1 - y0
    nx = abs(dx)
    ny = abs(dy)
    sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    sy = 1 if dy > 0 else (-1 if dy < 0 else 0)

    x = x0
    y = y0
    yield (x, y)
    ix = 0
    iy = 0
    while ix < nx or iy < ny:
        # 下一次竖直/水平格线穿越的参数比较，均乘以 nx*ny 消去除法。
        vx = (2 * ix + 1) * ny
        vy = (2 * iy + 1) * nx
        if vx < vy:
            x += sx
            ix += 1
        elif vx > vy:
            y += sy
            iy += 1
        else:
            # 恰好穿过格角：两侧格与对角格都算擦过。
            yield (x + sx, y)
            yield (x, y + sy)
            x += sx
            y += sy
            ix += 1
            iy += 1
        yield (x, y)


class NavGrid:
    """二维地形网格，提供 A* 寻路与视线平滑。"""

    __slots__ = ("_w", "_h", "_stride", "_blocked", "_mult", "_min_mult")

    def __init__(self, grid):
        width, height, flat = _parse_grid(grid)
        stride = width + 2
        size = (height + 2) * stride

        # 带一圈墙哨兵的扁平数组：邻居扩展无需做越界判断。
        blocked = bytearray(b"\x01") * size
        mult = [1.0] * size
        min_mult = _INF
        for y in range(height):
            row_base = (y + 1) * stride + 1
            src = y * width
            for x in range(width):
                idx = row_base + x
                v = flat[src + x]
                if v == 1.0:
                    blocked[idx] = 1  # 墙
                else:
                    blocked[idx] = 0
                    m = 1.0 if v == 0.0 else v
                    mult[idx] = m
                    if m < min_mult:
                        min_mult = m

        self._w = width
        self._h = height
        self._stride = stride
        # bytes / tuple 只读，构造后不再修改，保证并发安全。
        self._blocked = bytes(blocked)
        self._mult = tuple(mult)
        self._min_mult = 1.0 if min_mult is _INF else min_mult

    # ------------------------------------------------------------------
    # 校验辅助
    # ------------------------------------------------------------------
    def _point(self, point, what):
        if isinstance(point, (str, bytes, bytearray)):
            raise ValueError("%s must be an (x, y) pair" % what)
        try:
            x, y = point
        except (TypeError, ValueError):
            raise ValueError("%s must be an (x, y) pair" % what) from None
        if type(x) is not int or type(y) is not int:
            raise ValueError("%s coordinates must be integers" % what)
        if not (0 <= x < self._w and 0 <= y < self._h):
            raise ValueError("%s coordinate out of bounds" % what)
        return x, y

    def _index(self, x, y):
        return (y + 1) * self._stride + (x + 1)

    # ------------------------------------------------------------------
    # 寻路
    # ------------------------------------------------------------------
    def astar(self, start, goal):
        """返回从 ``start`` 到 ``goal`` 的最优路径（格坐标列表）。

        路径包含起点与终点，相邻点八邻域相连且不贴角穿墙；总代价是用
        地形倍率加权后的 Dijkstra 最优代价（对角步长 ``sqrt(2)``）。
        起点或终点越界抛 ``ValueError``；起点或终点是墙、或两者不连通
        时返回 ``None``。
        """
        sx, sy = self._point(start, "start")
        gx, gy = self._point(goal, "goal")

        stride = self._stride
        blocked = self._blocked
        mult = self._mult
        min_mult = self._min_mult

        sid = (sy + 1) * stride + (sx + 1)
        gid = (gy + 1) * stride + (gx + 1)
        if blocked[sid] or blocked[gid]:
            return None
        if sid == gid:
            return [(sx, sy)]

        size = len(blocked)
        gs = [_INF] * size          # 起点到此格的已知最优代价
        parent = [-1] * size
        closed = bytearray(size)    # 1 表示已以最优代价弹出
        gs[sid] = 0.0

        # 启发式按行/列分离：h = (max(dx, dy) + (sqrt2-1) * min(dx, dy)) * m。
        # dx 只依赖列、dy 只依赖行，故各存一张一维表，展开邻居时只做查表。
        ax = [abs(c - 1 - gx) for c in range(stride)]
        ay = [abs(r - 1 - gy) for r in range(self._h + 2)]

        def h_of(ncol, nrow):
            dx = ax[ncol]
            dy = ay[nrow]
            if dx > dy:
                return (dx + _OCTILE_K * dy) * min_mult
            return (dy + _OCTILE_K * dx) * min_mult

        # 平局时优先 h 更小（更靠近终点）的节点，显著减少开阔地上的扩展。
        h0 = h_of(sx + 1, sy + 1)
        heap = [(h0, h0, sid)]
        heappush = heapq.heappush
        heappop = heapq.heappop
        sqrt2 = _SQRT2

        while heap:
            f, h, node = heappop(heap)
            if closed[node] or f > gs[node] + h + _EPS:
                continue  # 已关闭，或过期条目
            if node == gid:
                path = []
                n = node
                while n != sid:
                    row, col = divmod(n, stride)
                    path.append((col - 1, row - 1))
                    n = parent[n]
                path.append((sx, sy))
                path.reverse()
                return path

            closed[node] = 1  # 启发式一致，首次弹出即最优
            base = gs[node]
            row, col = divmod(node, stride)

            # 正交四邻（逐个内联展开，热路径上省去循环开销）。
            nb = node - 1
            if not blocked[nb]:
                ng = base + mult[nb]
                if ng < gs[nb] and not closed[nb]:
                    gs[nb] = ng
                    parent[nb] = node
                    hh = h_of(col - 1, row)
                    heappush(heap, (ng + hh, hh, nb))
            nb = node + 1
            if not blocked[nb]:
                ng = base + mult[nb]
                if ng < gs[nb] and not closed[nb]:
                    gs[nb] = ng
                    parent[nb] = node
                    hh = h_of(col + 1, row)
                    heappush(heap, (ng + hh, hh, nb))
            nb = node - stride
            if not blocked[nb]:
                ng = base + mult[nb]
                if ng < gs[nb] and not closed[nb]:
                    gs[nb] = ng
                    parent[nb] = node
                    hh = h_of(col, row - 1)
                    heappush(heap, (ng + hh, hh, nb))
            nb = node + stride
            if not blocked[nb]:
                ng = base + mult[nb]
                if ng < gs[nb] and not closed[nb]:
                    gs[nb] = ng
                    parent[nb] = node
                    hh = h_of(col, row + 1)
                    heappush(heap, (ng + hh, hh, nb))

            # 对角四邻：两侧正交格都必须可通行，不能贴角穿墙。
            nb = node - stride - 1
            if not blocked[nb] and not blocked[node - 1] and not blocked[node - stride]:
                ng = base + sqrt2 * mult[nb]
                if ng < gs[nb] and not closed[nb]:
                    gs[nb] = ng
                    parent[nb] = node
                    hh = h_of(col - 1, row - 1)
                    heappush(heap, (ng + hh, hh, nb))
            nb = node - stride + 1
            if not blocked[nb] and not blocked[node + 1] and not blocked[node - stride]:
                ng = base + sqrt2 * mult[nb]
                if ng < gs[nb] and not closed[nb]:
                    gs[nb] = ng
                    parent[nb] = node
                    hh = h_of(col + 1, row - 1)
                    heappush(heap, (ng + hh, hh, nb))
            nb = node + stride - 1
            if not blocked[nb] and not blocked[node - 1] and not blocked[node + stride]:
                ng = base + sqrt2 * mult[nb]
                if ng < gs[nb] and not closed[nb]:
                    gs[nb] = ng
                    parent[nb] = node
                    hh = h_of(col - 1, row + 1)
                    heappush(heap, (ng + hh, hh, nb))
            nb = node + stride + 1
            if not blocked[nb] and not blocked[node + 1] and not blocked[node + stride]:
                ng = base + sqrt2 * mult[nb]
                if ng < gs[nb] and not closed[nb]:
                    gs[nb] = ng
                    parent[nb] = node
                    hh = h_of(col + 1, row + 1)
                    heappush(heap, (ng + hh, hh, nb))

        return None

    # ------------------------------------------------------------------
    # 平滑
    # ------------------------------------------------------------------
    def _los_bound(self, p, q):
        """返回 ``(长度, 线段所擦遍的最大地形倍率)``；擦到墙返回 ``None``。

        倍率取线段擦过的所有格（含两端点）的最大值。用它算出的
        ``长度 × 最大倍率`` 是这一段在“进入格计费”与“离开格计费”两种
        口径下的代价上界，因此据此做保守判断可保证平滑后代价不增加。
        """
        x0, y0 = p
        x1, y1 = q
        stride = self._stride
        blocked = self._blocked
        mult = self._mult

        max_mult = 0.0
        for x, y in _supercover(x0, y0, x1, y1):
            idx = (y + 1) * stride + (x + 1)
            if blocked[idx]:
                return None
            m = mult[idx]
            if m > max_mult:
                max_mult = m
        return math.hypot(x1 - x0, y1 - y0), max_mult

    def smooth(self, path):
        """用超覆盖视线把路径拉直，返回新的路径列表。

        逐段贪心地取“视线无碰撞且代价不增加”的最远点作为下一锚点，
        因此结果点数不多于输入、每段都无碰撞、总代价不大于输入。
        输入路径的每一段也必须是无碰撞的，否则抛 ``ValueError``。
        空路径返回空列表。
        """
        if isinstance(path, (str, bytes, bytearray)):
            raise ValueError("path must be a sequence of (x, y) points")
        try:
            raw = list(path)
        except TypeError:
            raise ValueError("path must be a sequence of (x, y) points") from None
        if not raw:
            return []

        pts = [self._point(p, "path point") for p in raw]
        stride = self._stride
        blocked = self._blocked
        mult = self._mult

        for x, y in pts:
            if blocked[(y + 1) * stride + (x + 1)]:
                raise ValueError("path point (%d, %d) is a wall" % (x, y))

        n = len(pts)
        if n == 1:
            return pts

        # 输入路径的每一段都必须无碰撞，否则无法定义“平滑后仍无碰撞”。
        enter = [0.0] * n   # 前缀代价：每步计入目标格的倍率
        leave = [0.0] * n   # 前缀代价：每步计入源格的倍率
        for k in range(n - 1):
            x0, y0 = pts[k]
            x1, y1 = pts[k + 1]
            if self._los_bound(pts[k], pts[k + 1]) is None:
                raise ValueError("path segment %d collides with an obstacle" % k)
            length = math.hypot(x1 - x0, y1 - y0)
            enter[k + 1] = enter[k] + length * mult[(y1 + 1) * stride + (x1 + 1)]
            leave[k + 1] = leave[k] + length * mult[(y0 + 1) * stride + (x0 + 1)]

        out = [pts[0]]
        i = 0
        limit = n - 1
        while i < limit:
            # 从锚点 i 向远处找“视线无碰撞且代价不增加”的最远点。
            # 先用指数探测定出 [可接受, 不可接受] 的区间，再二分收敛；
            # 每次判定都用真实视线与保守代价上界，所以搜索策略只影响
            # 拉直程度，不影响“无碰撞 / 代价不增加”的保证。
            lo = i + 1  # 单步等价原路径的那一段，总是可用
            step = 1
            hi = limit + 1
            while lo + step <= limit:
                j = lo + step
                if self._acceptable(i, j, pts, enter, leave):
                    lo = j
                    step *= 2
                else:
                    hi = j
                    step = 0x7FFFFFFF  # 跳出探测循环
            if hi > limit:
                hi = limit + 1
            # 在 (lo, hi) 内二分：lo 可接受、hi 不可接受（或越界）。
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if self._acceptable(i, mid, pts, enter, leave):
                    lo = mid
                else:
                    hi = mid
            out.append(pts[lo])
            i = lo
        return out

    def _acceptable(self, i, j, pts, enter, leave):
        """``pts[i] -> pts[j]`` 是否可替代原路径的同段子路径。"""
        bound = self._los_bound(pts[i], pts[j])
        if bound is None:
            return False
        seg = bound[0] * bound[1]
        return (seg <= (enter[j] - enter[i]) + _EPS
                and seg <= (leave[j] - leave[i]) + _EPS)
