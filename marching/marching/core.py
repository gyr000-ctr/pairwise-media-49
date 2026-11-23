"""Marching Squares 等值线提取。

坐标约定：``field[y][x]`` 是网格顶点 ``(x, y)`` 处的值，``x`` 向右、
``y`` 向下；单元 ``(x, y)`` 由四个相邻顶点围成。返回的每条折线都是
闭合的（首尾点相同），并且所有点都落在网格范围内。

分类采用 ``>= level`` 为实心。鞍点单元（对角两实心、对角两空）按双
线性插值的中心值 ``(tl + tr + br + bl) / 4`` 判定连接方式：中心为实心
时两个实心角连通，等值线绕开空角；中心为空时两个空角连通，等值线绕
开实心角。中心值恰好等于 level 时按实心处理，因此等值线永远不会横穿
实心区。

顶点值恰好等于 level 时，插值点会退化为顶点本身；实现给每条网格边
分配独立节点并把插值点向内收一个极小量，避免相邻边节点重合造成度数
异常。由此产生的零面积退化碎片会被丢弃。

跨越网格边界的等值线沿边界的"高值"弧段闭合，因此结果始终是闭合折
线。实现不使用任何实例状态，可并发调用。
"""

import math

# 单元四条边（以角点命名的方向标签）
_T, _R, _B, _L = 0, 1, 2, 3

# 非鞍点情形下，每条等值线段连接的两条边。
# bits: TL=1, TR=2, BR=4, BL=8（位为 1 表示该角点 >= level）
_PAIRS = {
    1: ((_T, _L),),
    2: ((_T, _R),),
    3: ((_L, _R),),
    4: ((_R, _B),),
    6: ((_T, _B),),
    7: ((_L, _B),),
    8: ((_B, _L),),
    9: ((_T, _B),),
    11: ((_R, _B),),
    12: ((_R, _L),),
    13: ((_T, _R),),
    14: ((_T, _L),),
}


class InvalidInput(ValueError, TypeError):
    """非法输入。同时是 ``ValueError`` 与 ``TypeError``，便于调用方捕获。"""


def _to_float(value, what):
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        raise InvalidInput("%s 必须是实数，收到 %s" % (what, type(value).__name__))
    try:
        fv = float(value)
    except (TypeError, ValueError, OverflowError):
        raise InvalidInput("%s 必须是实数，收到 %s"
                           % (what, type(value).__name__)) from None
    if not math.isfinite(fv):
        raise InvalidInput("%s 不能包含 NaN 或无穷大" % what)
    return fv


def _validate_field(field):
    """校验并展平网格，返回 ``(flat_values, height, width)``。"""
    if isinstance(field, (str, bytes, bytearray)) or field is None:
        raise InvalidInput("field 必须是二维数值序列")
    try:
        rows = list(field)
    except TypeError:
        raise InvalidInput("field 必须是二维数值序列") from None

    if not rows:
        return [], 0, 0

    flat = []
    width = None
    for row in rows:
        if isinstance(row, (str, bytes, bytearray)) or row is None:
            raise InvalidInput("field 的每一行都必须是数值序列")
        try:
            cells = list(row)
        except TypeError:
            raise InvalidInput("field 的每一行都必须是数值序列") from None
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise InvalidInput("field 的各行长度必须一致")
        for v in cells:
            flat.append(_to_float(v, "field"))

    if width == 0:
        return [], 0, 0
    return flat, len(rows), width


def _validate_level(level):
    return _to_float(level, "level")


class Marching:
    def contour(self, field, level):
        """提取 ``level`` 等值线，返回闭合折线列表。

        每条折线是 ``[(x, y), ...]``，首尾点相同。没有等值线时返回 ``[]``。
        非法输入（非数值、非有限值、参差的网格等）抛出 ``ValueError``。
        """
        values, h, w = _validate_field(field)
        lv = _validate_level(level)

        nw = w - 1
        nh = h - 1
        if nw <= 0 or nh <= 0:
            # 没有完整的单元，也就不存在等值线
            return []

        # 节点编号按"网格边"分配：
        #   水平边 (y, x) -> y * nw + x
        #   垂直边 (y, x) -> h_cnt + y * w + x
        # 每条边有独立的节点，即使插值点恰好落在顶点上也不会与相邻边
        # 合并成同一点，从而保证每个节点的度数不超过 2。
        h_cnt = h * nw
        v_base = h_cnt
        total_nodes = v_base + nh * w

        points = [None] * total_nodes   # 节点 -> 坐标
        # 等值点严格落在边的内部：顶点恰好等于 level（或浮点误差使其
        # 恰好落在端点）时向内收缩一点，避免相邻边的等值点重合导致
        # 节点度数异常、折线被错误拼接。
        eps = 1e-12

        def _t(va, vb):
            t = (lv - va) / (vb - va)
            if t < eps:
                return eps
            if t > 1.0 - eps:
                return 1.0 - eps
            return t

        def h_node(y, x):
            eid = y * nw + x
            if points[eid] is None:
                va = values[y * w + x]
                vb = values[y * w + x + 1]
                points[eid] = (x + _t(va, vb), float(y))
            return eid

        def v_node(y, x):
            eid = v_base + y * w + x
            if points[eid] is None:
                va = values[y * w + x]
                vb = values[(y + 1) * w + x]
                points[eid] = (float(x), y + _t(va, vb))
            return eid

        # 每条边记录为 (kind, a, b, arc_points)
        # kind 0: 等值线段；kind 1: 边界的闭合弧段
        edges = []
        adj = {}

        def add_edge(kind, a, b, arc=None):
            eid = len(edges)
            edges.append((kind, a, b, arc))
            adj.setdefault(a, []).append(eid)
            adj.setdefault(b, []).append(eid)

        for y in range(nh):
            row = y * w
            next_row = row + w
            for x in range(nw):
                va = values[row + x]          # TL
                vb = values[row + x + 1]      # TR
                vc = values[next_row + x + 1]  # BR
                vd = values[next_row + x]      # BL
                bits = ((1 if va >= lv else 0)
                        | (2 if vb >= lv else 0)
                        | (4 if vc >= lv else 0)
                        | (8 if vd >= lv else 0))
                if bits == 0 or bits == 15:
                    continue
                if bits == 5 or bits == 10:
                    center = (va + vb + vc + vd) * 0.25
                    if bits == 5:
                        if center >= lv:
                            pairs = ((_T, _R), (_B, _L))
                        else:
                            pairs = ((_T, _L), (_R, _B))
                    else:
                        if center >= lv:
                            pairs = ((_T, _L), (_R, _B))
                        else:
                            pairs = ((_T, _R), (_B, _L))
                else:
                    pairs = _PAIRS[bits]

                for e1, e2 in pairs:
                    n1 = _cell_node(e1, y, x, h_node, v_node)
                    n2 = _cell_node(e2, y, x, h_node, v_node)
                    if n1 != n2:
                        add_edge(0, n1, n2)

        if not edges:
            return []

        # ---- 边界闭合：把边界上的开放端点用高值弧段连接起来 ----
        perimeter = 2 * nw + 2 * nh

        def is_boundary(pt):
            x, y = pt
            return x == 0.0 or x == nw or y == 0.0 or y == nh

        def perim_pos(pt):
            x, y = pt
            if y == 0.0:
                return x
            if x == nw:
                return nw + y
            if y == nh:
                return nw + nh + (nw - x)
            return nw + nh + nw + (nh - y)

        def perim_pt(k):
            k %= perimeter
            if k <= nw:
                return (float(k), 0.0)
            k -= nw
            if k <= nh:
                return (float(nw), float(k))
            k -= nh
            if k <= nw:
                return (float(nw - k), float(nh))
            k -= nw
            return (0.0, float(nh - k))

        def perim_value(p):
            p %= perimeter
            if p <= nw:
                xf = p
                if xf >= nw:
                    return values[nw]
                x0 = int(xf)
                t = xf - x0
                return values[x0] * (1.0 - t) + values[x0 + 1] * t
            p -= nw
            if p <= nh:
                yf = p
                if yf >= nh:
                    return values[nh * w + nw]
                y0 = int(yf)
                t = yf - y0
                a = values[y0 * w + nw]
                b = values[(y0 + 1) * w + nw]
                return a * (1.0 - t) + b * t
            p -= nh
            if p <= nw:
                xf = nw - p
                if xf <= 0.0:
                    return values[nh * w]
                x0 = int(xf)
                t = xf - x0
                row = nh * w
                return values[row + x0] * (1.0 - t) + values[row + x0 + 1] * t
            p -= nw
            yf = nh - p
            if yf <= 0.0:
                return values[0]
            y0 = int(yf)
            t = yf - y0
            return values[y0 * w] * (1.0 - t) + values[(y0 + 1) * w] * t

        crossings = []
        for node, eids in adj.items():
            if len(eids) != 1:
                continue
            pt = points[node]
            if is_boundary(pt):
                crossings.append((perim_pos(pt), node))

        if len(crossings) >= 2:
            crossings.sort()
            m = len(crossings)
            for i in range(m):
                p0, n0 = crossings[i]
                p1, n1 = crossings[(i + 1) % m]
                length = p1 - p0
                if length <= 0.0:
                    length += perimeter
                if length <= 0.0:
                    continue
                mid = (p0 + length * 0.5) % perimeter
                if perim_value(mid) < lv:
                    continue
                # 沿边界（周长参数递增方向）从 n0 走到 n1。
                # 只保存内部点：两端点由折线自身的节点坐标提供。
                end = p0 + length
                arc = []
                k = math.floor(p0) + 1
                while k < end - 1e-12:
                    q = perim_pt(k % perimeter)
                    if not arc or arc[-1] != q:
                        arc.append(q)
                    k += 1
                add_edge(1, n0, n1, arc)

        # ---- 沿边链行走，拼出闭合折线 ----
        used = [False] * len(edges)
        result = []
        for eid0 in range(len(edges)):
            if used[eid0]:
                continue
            used[eid0] = True
            kind, a, b, arc = edges[eid0]
            start = a
            if kind == 0:
                path = [points[a], points[b]]
            else:
                path = [points[a]]
                path.extend(arc)
                path.append(points[b])
            cur = b
            while cur != start:
                nxt = None
                for cand in adj[cur]:
                    if not used[cand]:
                        nxt = cand
                        break
                if nxt is None:
                    break
                used[nxt] = True
                kind, a, b, arc = edges[nxt]
                other = b if a == cur else a
                if kind == 0:
                    path.append(points[other])
                else:
                    if a == cur:
                        path.extend(arc)
                    else:
                        path.extend(reversed(arc))
                    path.append(points[other])
                cur = other

            if cur != start:
                path.append(points[start])

            # 清理重复点：等值点被 eps 推离顶点后，相邻边的等值点可能
            # 相差一个极小距离，按容差视为同一点。
            tol = _DEDUP_TOL
            clean = []
            for p in path:
                if not clean or _far(clean[-1], p, tol):
                    clean.append(p)
            while len(clean) >= 2 and not _far(clean[0], clean[-1], tol):
                clean.pop()
            if len(clean) < 3:
                continue
            if _polygon_area(clean) <= _MIN_AREA:
                # 顶点恰好落在等值面上时会退化成几乎零面积的碎片，丢弃
                continue
            clean.append(clean[0])
            result.append(clean)

        return result


# 退化碎片（面积 ~ eps）与真实等值环（面积通常 >= 1e-2）之间的阈值
_MIN_AREA = 1e-9
# 被视为同一点的距离阈值（远大于 eps，远小于一个单元）
_DEDUP_TOL = 1e-8


def _far(p, q, tol):
    return abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol


def _polygon_area(loop):
    total = 0.0
    for (x0, y0), (x1, y1) in zip(loop, loop[1:]):
        total += x0 * y1 - x1 * y0
    x0, y0 = loop[-1]
    x1, y1 = loop[0]
    total += x0 * y1 - x1 * y0
    return abs(total) * 0.5


def _cell_node(edge, y, x, h_node, v_node):
    if edge == _T:
        return h_node(y, x)
    if edge == _R:
        return v_node(y, x + 1)
    if edge == _B:
        return h_node(y + 1, x)
    return v_node(y, x)  # _L
