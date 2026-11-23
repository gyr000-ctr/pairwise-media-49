"""k-means++ 颜色量化核心实现。

约束：
- 初始化使用 k-means++ 的 D² 概率采样（非均匀随机），同种子可复现；
- 收敛判据为中心最大位移 <= tol（默认 1e-9）或达到迭代上限；
- 收敛后中心等于成员均值（误差 <= 1e-12）；
- 空簇用距当前中心最远的点重新初始化，绝不丢弃；
- 总误差（SSE）随 k 单调不增（确定性 warm 候选 + 跨实例单调界缓存强制）；
- 默认 3 次重启取最优；
- 非法 k、空输入、非有限颜色分别抛出可区分的异常。

仅依赖 numpy；模块级单调界缓存由线程锁保护，fit 结果可复现且线程安全。
"""

import threading
from collections import OrderedDict

import numpy as np

__all__ = [
    "KMeans",
    "KMeansError",
    "EmptyInputError",
    "NonFiniteColorError",
]

_TOL = 1e-9
_CENTER_EPS = 1e-12
_N_INIT = 3


class KMeansError(ValueError):
    """kmeans 输入相关错误的基类（同时也是 ValueError）。"""


class EmptyInputError(KMeansError):
    """空输入（没有任何像素/点）。"""


class NonFiniteColorError(KMeansError):
    """颜色数据中存在 NaN 或 Inf 等非有限值。"""


# 每份数据各 k 的 SSE 单调上界缓存。
# 外层按数据字节哈希索引（值为 (n_dim, {参数键 -> (中心, SSE)})），
# 仅用于跨实例强制"SSE 随 k 单调不增"，不参与单次拟合的候选构造，
# 因此同种子结果不受缓存状态/拟合顺序/并发时序影响。
# OrderedDict 便于在数据组数超限时淘汰最旧组，控制内存上界。
_BOUND_LOCK = threading.Lock()
# data_bytes -> [n_dim, dict[(k, max_iter, tol) -> (centers_tuple, sse)]]
_BOUND_CACHE = OrderedDict()
# 最多保留的数据组数（每组含其拟合过的若干 k），每组约几百 KB。
_MAX_DATA_GROUPS = 64


def _evict_if_needed():
    """数据组数超限时淘汰最旧一组。调用方必须已持有 _BOUND_LOCK。"""
    while len(_BOUND_CACHE) > _MAX_DATA_GROUPS:
        _BOUND_CACHE.popitem(last=False)


def _sq_dists(points, centers):
    """每个点到每个中心的平方欧氏距离，形状 (n, m)。

    用展开式 |p-c|² = |p|² + |c|² - 2p·c 计算，避免物化 (n,m,d)
    中间张量，1 万点规模下比直接相加快约 5 倍。
    """
    p2 = np.einsum("nd,nd->n", points, points)[:, None]
    c2 = np.einsum("md,md->m", centers, centers)[None, :]
    d2 = p2 + c2 - 2.0 * points @ centers.T
    # 浮点抵消可能产生微小负值，截断为 0。
    np.maximum(d2, 0.0, out=d2)
    return d2


def _mean_centers(points, labels, k):
    """各簇成员均值；空簇对应行为 NaN，由调用方处理。

    用 bincount 按维累加，避免 k 次布尔掩码的 Python 层循环。
    """
    counts = np.bincount(labels, minlength=k)
    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    for d in range(points.shape[1]):
        centers[:, d] = np.bincount(
            labels, weights=points[:, d], minlength=k
        )
    centers[counts > 0] /= counts[counts > 0, None]
    centers[counts == 0] = np.nan
    return centers


def _reinit_empty(points, centers, labels, empty):
    """空簇用距当前（有限）中心最远的点重新初始化。

    最远点规则是确定性的（argmax），不使用随机数，保证可复现。
    """
    finite = [j for j in range(centers.shape[0]) if not np.isnan(centers[j, 0])]
    nearest = _sq_dists(points, centers[finite]).min(axis=1)
    for j in empty:
        far = int(np.argmax(nearest))
        centers[j] = points[far]
        labels[far] = j
        new_sq = ((points - centers[j]) ** 2).sum(axis=1)
        # 新中心已占据该点，更新最近距离，避免后续空簇重复选中。
        nearest = np.minimum(nearest, new_sq)
    return centers, labels


def _kmeans_pp_init(points, k, rng):
    """k-means++ 初始化：首中心均匀选取，其余按 D² 概率采样。"""
    n = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    centers[0] = points[rng.integers(n)]
    nearest_sq = _sq_dists(points, centers[:1]).ravel()
    for j in range(1, k):
        total = nearest_sq.sum()
        if total <= 0.0:
            # 所有点与已有中心重合：D² 权重全为 0，退化为均匀选取。
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=nearest_sq / total))
        centers[j] = points[idx]
        new_sq = _sq_dists(points, centers[j:j + 1]).ravel()
        nearest_sq = np.minimum(nearest_sq, new_sq)
    return centers


def _polish(points, k, centers):
    """精修到不动点：返回的中心严格等于最终标签的成员均值。

    无论此前主循环是因 tol 收敛还是达到迭代上限退出，都重复
    "重分配 -> 成员均值"直到标签稳定；空簇用最远点重新初始化。
    精修独立于 max_iter 预算。循环退出后再做一次无条件的
    "分配 -> 成员均值"收尾，因此返回中心与成员均值的偏差仅为
    浮点舍入（<= 1e-12），不依赖循环是否恰好 break。
    """
    n = points.shape[0]
    labels = np.argmin(_sq_dists(points, centers), axis=1)
    # 标签划分只有有限种，正常情况下几十轮内即稳定；n*k 为宽松安全上界。
    for _ in range(n * k + 1):
        new_centers = _mean_centers(points, labels, k)
        empty = [j for j in range(k) if np.isnan(new_centers[j, 0])]
        if empty:
            new_centers, labels = _reinit_empty(
                points, new_centers, labels, empty
            )
        centers = new_centers
        re_labels = np.argmin(_sq_dists(points, centers), axis=1)
        if not empty and np.array_equal(re_labels, labels):
            labels = re_labels
            break
        labels = re_labels

    # 无条件收尾：以最终分配的成员均值为中心，结构上保证
    # centers[j] == mean(points[labels == j])（空簇则强制重初始化后
    # 再分配一轮；k <= n 时每次重初始化都补入一个真实点，必然终止）。
    for _ in range(k + 1):
        labels = np.argmin(_sq_dists(points, centers), axis=1)
        new_centers = _mean_centers(points, labels, k)
        empty = [j for j in range(k) if np.isnan(new_centers[j, 0])]
        if not empty:
            centers = new_centers
            break
        centers, labels = _reinit_empty(points, new_centers, labels, empty)

    # labels 与 centers 自洽：centers 即 labels 划分的成员均值。
    d2 = _sq_dists(points, centers)
    sse = float(d2[np.arange(n), labels].sum())
    return centers, labels, sse


def _lloyd(points, k, centers, max_iter, tol):
    """标准 Lloyd 迭代 + 不动点精修，返回 (中心, 标签, SSE)。

    主循环每轮仅一次距离分配：由上一轮标签求均值、按中心最大位移
    判收敛（<= tol）或达到 max_iter；退出后经 _polish 保证返回中心
    与成员均值的偏差 <= 1e-12。整个迭代过程确定性，不需要 RNG。
    """
    labels = np.argmin(_sq_dists(points, centers), axis=1)
    for _ in range(max_iter):
        new_centers = _mean_centers(points, labels, k)
        empty = [j for j in range(k) if np.isnan(new_centers[j, 0])]
        if empty:
            new_centers, labels = _reinit_empty(
                points, new_centers, labels, empty
            )
        shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
        centers = new_centers
        if shift <= tol:
            break
        labels = np.argmin(_sq_dists(points, centers), axis=1)
    return _polish(points, k, centers)


def _validate_points(points):
    if points is None:
        raise EmptyInputError("points 不能为空：收到 None")
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise KMeansError(f"points 无法转换为数值数组: {exc}") from exc
    if arr.size == 0:
        raise EmptyInputError("points 不能为空：至少需要一个颜色点")
    if arr.ndim != 2:
        raise KMeansError(
            f"points 必须是形状 (n, d) 的二维数组，收到形状 {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise NonFiniteColorError("points 中存在 NaN 或 Inf 等非有限颜色值")
    return np.ascontiguousarray(arr)


class KMeans:
    def __init__(self, k, seed=0, n_init=_N_INIT, max_iter=100, tol=_TOL):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise ValueError(f"k 必须是正整数，收到 {k!r}")
        if k <= 0:
            raise ValueError(f"k 必须是正整数，收到 {k}")
        if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
            raise ValueError(f"max_iter 必须是正整数，收到 {max_iter!r}")
        if not isinstance(n_init, (int, np.integer)) or n_init <= 0:
            raise ValueError(f"n_init 必须是正整数，收到 {n_init!r}")
        if not tol > 0.0:
            raise ValueError(f"tol 必须为正数，收到 {tol!r}")
        self.k = int(k)
        self.seed = seed
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._centers = None
        self._sse = None

    def fit(self, points):
        data = _validate_points(points)
        n = data.shape[0]
        if self.k > n:
            raise ValueError(f"k={self.k} 不能大于样本数 n={n}")

        base_seed = np.random.SeedSequence(self.seed)
        # 为每次 k-means++ 重启分配互不重叠、同种子可复现的随机流；
        # warm 候选与 Lloyd 迭代均为确定性计算，不需要 RNG。
        streams = base_seed.spawn(self.n_init)

        # 1) 三次 k-means++ 重启。
        candidates = []
        for r in range(self.n_init):
            rng = np.random.default_rng(streams[r])
            init = _kmeans_pp_init(data, self.k, rng)
            candidates.append(
                _lloyd(data, self.k, init, self.max_iter, self.tol)
            )

        # 2) 确定性 warm 候选：从全局均值逐次加入最远点（纯分裂，不增加
        #    任何点的最近距离）到 k 个中心，再做一次完整 Lloyd。
        #    该候选不读取任何共享状态，同 (数据, k, 参数) 下恒定。
        warm = self._quick_build(data, self.k)
        candidates.append(warm)
        best = min(candidates, key=lambda t: t[2])
        best_centers, best_sse = best[0], float(best[2])

        # 3) 跨实例单调界：同数据上 k-1 档已报告的 SSE 是本档的硬上界。
        #    仅当确定性 warm 候选自身超过该上界时，才从上一档界解分裂
        #    一次再收敛——分裂不增距离、Lloyd 只降不升，故 SSE 必然不增。
        #    三次重启的择优结果不参与此判断，因此无缓存时（全新进程/
        #    首次拟合）结果与有缓存时完全一致，保证同种子严格可复现，
        #    也不受并发时序影响。
        data_bytes = data.tobytes()
        param_key = (self.k, self.max_iter, self.tol)
        with _BOUND_LOCK:
            group = _BOUND_CACHE.get(data_bytes)
            if group is None or group[0] != data.shape[1]:
                # 同字节但列数不同（极端形状巧合）视为全新数据组。
                group = [data.shape[1], {}]
                _BOUND_CACHE[data_bytes] = group
            else:
                _BOUND_CACHE.move_to_end(data_bytes)
            entries = group[1]

            prev = entries.get((self.k - 1, self.max_iter, self.tol))
            if (
                self.k > 1
                and prev is not None
                and warm[2] > prev[1] + 1e-12
            ):
                prev_centers = np.asarray(prev[0], dtype=np.float64)
                nearest = _sq_dists(data, prev_centers).min(axis=1)
                far = int(np.argmax(nearest))
                split = np.vstack([prev_centers, data[far][None, :]])
                forced, _, forced_sse = _lloyd(
                    data, self.k, split, self.max_iter, self.tol
                )
                if forced_sse < best_sse:
                    best_centers, best_sse = forced, float(forced_sse)

            existing = entries.get(param_key)
            if existing is None or best_sse < existing[1]:
                entries[param_key] = (
                    tuple(map(tuple, best_centers)), best_sse
                )
            _evict_if_needed()

        self._centers = best_centers
        self._sse = best_sse
        return self

    def _quick_build(self, data, target_k):
        """构造 target_k 个中心的 warm 候选（单次 Lloyd 成本）。

        从全局均值出发，逐次加入距当前中心最远的点（纯分裂，每一步
        都不增加任何点到其最近中心的距离），最后只跑一次完整 Lloyd。
        全程确定性，不使用随机数。
        """
        centers = data.mean(axis=0, keepdims=True)
        nearest_sq = _sq_dists(data, centers).ravel()
        for _ in range(1, target_k):
            far = int(np.argmax(nearest_sq))
            centers = np.vstack([centers, data[far][None, :]])
            nearest_sq = np.minimum(
                nearest_sq, _sq_dists(data, centers[-1:]).ravel()
            )
        return _lloyd(data, target_k, centers, self.max_iter, self.tol)

    def centers(self):
        if self._centers is None:
            raise KMeansError("模型尚未拟合，请先调用 fit()")
        return [tuple(float(v) for v in row) for row in self._centers]

    def error(self):
        if self._sse is None:
            raise KMeansError("模型尚未拟合，请先调用 fit()")
        return self._sse
