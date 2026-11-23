"""bvh2 核心 API：场景 BVH 与最近命中查询。

坐标约定：盒由 ``(lo, hi)`` 两个三维点给出，盒表面包含端点；
射线由 ``origin + direction * t`` 参数化，``t >= 0``。

查询结果统一为 ``(t, name)``，``t`` 是射线进入命中盒对应的参数值
（起点在盒内时为 0.0）。同距离时取下标更小的盒，保证确定性。

实现要点：
- 建树用中位数切分（按质心方差最大的轴排序后对半分）；
- 查询用显式栈遍历，按进入距离先近后远，并用当前最优 ``t`` 剪枝；
- 建树完成后树结构只读，因此多线程并发查询是安全的。
"""

import math
import threading

__all__ = ["Bvh"]

_INF = math.inf
_LEAF_SIZE = 4
_EPS = 1e-12


def _as_vec(v, what):
    """把任意长度为 3 的序列规整成有限的 float 三元组。"""
    if isinstance(v, (str, bytes, bytearray)):
        raise ValueError("%s 必须是 3 个数值" % what)
    try:
        x, y, z = v
    except (TypeError, ValueError):
        raise ValueError("%s 必须是 3 个数值" % what) from None
    try:
        x = float(x)
        y = float(y)
        z = float(z)
    except (TypeError, ValueError):
        raise ValueError("%s 必须是 3 个数值" % what) from None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        raise ValueError("%s 不能包含非有限数值" % what)
    return (x, y, z)


def _as_box(box):
    """校验并规整 ``(lo, hi)``；反向盒或非法值抛 ValueError。"""
    if isinstance(box, (str, bytes, bytearray)):
        raise ValueError("盒必须是 (lo, hi) 两个三维点")
    try:
        lo, hi = box
    except (TypeError, ValueError):
        raise ValueError("盒必须是 (lo, hi) 两个三维点") from None
    lo = _as_vec(lo, "盒下界")
    hi = _as_vec(hi, "盒上界")
    if lo[0] > hi[0] or lo[1] > hi[1] or lo[2] > hi[2]:
        raise ValueError("盒下界不得大于上界")
    return lo, hi


def _slab(lo, hi, o, d):
    """射线与轴对齐盒求交，返回 ``(tmin, tmax)``；无交返回 None。

    ``tmin``/``tmax`` 可为负（盒在射线后方），由调用方判断。
    平行于某轴时检查原点是否落在该轴区间内（闭区间，端点算命中）。
    非有限值已在入口校验，故不会出现 inf * 0 之类的 NaN。
    """
    ox, oy, oz = o
    dx, dy, dz = d
    lx, ly, lz = lo
    hx, hy, hz = hi

    if dx == 0.0:
        if ox < lx or ox > hx:
            return None
        tmin = -_INF
        tmax = _INF
    else:
        inv = 1.0 / dx
        t1 = (lx - ox) * inv
        t2 = (hx - ox) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = t1
        tmax = t2

    if dy == 0.0:
        if oy < ly or oy > hy:
            return None
    else:
        inv = 1.0 / dy
        t1 = (ly - oy) * inv
        t2 = (hy - oy) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return None

    if dz == 0.0:
        if oz < lz or oz > hz:
            return None
    else:
        inv = 1.0 / dz
        t1 = (lz - oz) * inv
        t2 = (hz - oz) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return None

    if tmax < 0.0:
        return None
    return tmin, tmax


class _Node:
    """BVH 节点；叶子 ``left is None``，用 ``start``/``end`` 索引盒数组。"""

    __slots__ = ("lo", "hi", "left", "right", "start", "end")

    def __init__(self, lo, hi, left=None, right=None, start=0, end=0):
        self.lo = lo
        self.hi = hi
        self.left = left
        self.right = right
        self.start = start
        self.end = end


class Bvh:
    """场景 BVH。

    用法::

        b = Bvh()
        b.add("a", ((0, 0, 0), (1, 1, 1)))
        b.build()
        t, name = b.nearest((0.5, 0.5, -1.0), (0.0, 0.0, 1.0))
    """

    def __init__(self):
        self._lo = []
        self._hi = []
        self._names = []
        self._orig = []
        self._root = None
        self._built = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 构建
    def add(self, name, box):
        """加入一个盒；盒结构或数值不合法时抛 ValueError。"""
        lo, hi = _as_box(box)
        with self._lock:
            self._lo.append(lo)
            self._hi.append(hi)
            self._names.append(name)
            # 记录盒加入顺序，同距离命中时按它判定"下标更小"。
            self._orig.append(len(self._orig))
            # 新盒加入后原有树失效，需要重新 build()。
            self._built = False
            self._root = None
        return self

    def build(self):
        """建树。空场景也可建树，查询时再报错。"""
        with self._lock:
            n = len(self._lo)
            if n == 0:
                self._root = None
                self._built = True
                return self

            lo = self._lo
            hi = self._hi
            order = list(range(n))
            root = _build_node(order, 0, n, lo, hi)

            # 按叶子顺序重排数据，叶子用连续区间索引。
            self._lo = [lo[i] for i in order]
            self._hi = [hi[i] for i in order]
            self._names = [self._names[i] for i in order]
            self._orig = [self._orig[i] for i in order]
            self._root = root
            self._built = True
        return self

    # ------------------------------------------------------------------ 查询
    def nearest(self, origin, direction):
        """返回最近命中 ``(t, name)``；未命中返回 None。

        - 空场景（或尚未 build）抛 ValueError；
        - origin / direction 含非有限值或方向为零向量抛 ValueError。
        """
        o = _as_vec(origin, "origin")
        d = _as_vec(direction, "direction")
        if d == (0.0, 0.0, 0.0):
            raise ValueError("direction 不能是零向量")
        if not self._built:
            raise ValueError("尚未 build()，无法查询")
        root = self._root
        if root is None:
            raise ValueError("空场景，无法查询最近命中")

        lo = self._lo
        hi = self._hi
        names = self._names
        orig = self._orig

        best_t = _INF
        best_i = -1
        best_orig = -1
        stack = [root]
        push = stack.append
        pop = stack.pop

        while stack:
            node = pop()
            res = _slab(node.lo, node.hi, o, d)
            if res is None:
                continue
            tmin, tmax = res
            if tmin > best_t:
                continue
            if tmax < 0.0:
                continue

            left = node.left
            if left is None:
                for i in range(node.start, node.end):
                    r = _slab(lo[i], hi[i], o, d)
                    if r is None:
                        continue
                    t = r[0]
                    if t < 0.0:
                        t = 0.0
                    if t < best_t or (t == best_t and orig[i] < best_orig):
                        best_t = t
                        best_i = i
                        best_orig = orig[i]
                continue

            right = node.right
            lr = _slab(left.lo, left.hi, o, d)
            rr = _slab(right.lo, right.hi, o, d)
            if lr is not None and rr is not None:
                # 先压远的，近的后压先出栈。
                if lr[0] <= rr[0]:
                    push(right)
                    push(left)
                else:
                    push(left)
                    push(right)
            elif lr is not None:
                push(left)
            elif rr is not None:
                push(right)

        if best_i < 0:
            return None
        return (float(best_t), names[best_i])


def _segment_bounds(order, start, end, lo, hi):
    """返回区间内所有盒的包围盒与质心包围盒。"""
    mn0 = mn1 = mn2 = _INF
    mx0 = mx1 = mx2 = -_INF
    cmin0 = cmin1 = cmin2 = _INF
    cmax0 = cmax1 = cmax2 = -_INF
    for k in range(start, end):
        i = order[k]
        ax, ay, az = lo[i]
        bx, by, bz = hi[i]
        if ax < mn0:
            mn0 = ax
        if ay < mn1:
            mn1 = ay
        if az < mn2:
            mn2 = az
        if bx > mx0:
            mx0 = bx
        if by > mx1:
            mx1 = by
        if bz > mx2:
            mx2 = bz
        cx = (ax + bx) * 0.5
        cy = (ay + by) * 0.5
        cz = (az + bz) * 0.5
        if cx < cmin0:
            cmin0 = cx
        if cy < cmin1:
            cmin1 = cy
        if cz < cmin2:
            cmin2 = cz
        if cx > cmax0:
            cmax0 = cx
        if cy > cmax1:
            cmax1 = cy
        if cz > cmax2:
            cmax2 = cz
    return (mn0, mn1, mn2), (mx0, mx1, mx2), \
        (cmin0, cmin1, cmin2), (cmax0, cmax1, cmax2)


def _build_node(order, start, end, lo, hi):
    """递归建树：区间内按质心方差最大轴中位数切分，最多 _LEAF_SIZE 个成叶。"""
    n = end - start
    mn, mx, cmin, cmax = _segment_bounds(order, start, end, lo, hi)

    if n <= _LEAF_SIZE:
        return _Node(mn, mx, start=start, end=end)

    ext0 = cmax[0] - cmin[0]
    ext1 = cmax[1] - cmin[1]
    ext2 = cmax[2] - cmin[2]
    if ext0 >= ext1 and ext0 >= ext2:
        axis = 0
        ext = ext0
    elif ext1 >= ext2:
        axis = 1
        ext = ext1
    else:
        axis = 2
        ext = ext2

    # 质心全重合时无法再切分，直接成叶。
    if ext <= _EPS:
        return _Node(mn, mx, start=start, end=end)

    order[start:end] = sorted(
        order[start:end],
        key=lambda i: (lo[i][axis] + hi[i][axis]) * 0.5,
    )
    mid = start + n // 2
    left = _build_node(order, start, mid, lo, hi)
    right = _build_node(order, mid, end, lo, hi)
    return _Node(mn, mx, left=left, right=right)
