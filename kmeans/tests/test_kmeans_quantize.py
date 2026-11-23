"""k-means++ 颜色量化约束测试（新增，不改动既有 tests/test_core.py）。

覆盖题面全部硬约束：
1. D² 采样初始化、同种子可复现；
2. 收敛判据 tol=1e-9 / max_iter，收敛后中心 == 成员均值（<=1e-12）；
3. 空簇用最远点重新初始化，簇数不丢失；
4. SSE 随 k 单调不增（含跨实例、乱序拟合）；
5. 3 次重启取最优；
6. 1 万像素 8 色 <= 300 ms；
7. 非法 k / 空输入 / 非有限颜色可区分报错；
8. 线程安全。
"""

import threading

import numpy as np
import pytest

import kmeans
from kmeans import core as kmcore
from kmeans.core import EmptyInputError, NonFiniteColorError


def _blobs(n_per=400, seed=1):
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
         [1.0, 0.0, 0.2], [0.0, 1.0, 0.8]],
    )
    pts = np.vstack(
        [c + rng.normal(scale=0.03, size=(n_per, 3)) for c in centers]
    )
    return pts


# ---------- 1. k-means++ 的 D² 采样与可复现性 ----------

class _StubRng:
    """记录 choice 调用并返回固定下标，用于检验 D² 概率权重。"""

    def __init__(self, pick):
        self.pick = pick
        self.calls = []

    def integers(self, n):
        return 0

    def choice(self, n, p=None):
        self.calls.append(np.asarray(p).copy())
        return self.pick


def test_kmeans_pp_uses_squared_distance_weights():
    pts = np.array([[0.0], [1.0], [3.0], [10.0]])
    stub = _StubRng(pick=1)
    kmcore._kmeans_pp_init(pts, 2, stub)
    # 首中心固定为 pts[0]=0，D² 权重应正比于到最近中心的平方距离。
    expected = np.array([0.0, 1.0, 9.0, 100.0])
    expected = expected / expected.sum()
    assert stub.calls, "第二个中心必须经过加权 choice 采样，而非均匀随机"
    np.testing.assert_allclose(stub.calls[0], expected, rtol=1e-12)


def test_kmeans_pp_not_uniform_sampling():
    # 三个重合零点 + 一个远点：D² 采样下远点成为第二中心的概率
    # 应为 3/4（首中心落在零点时权重 100% 指向远点），
    # 显著高于均匀随机的 1/4。
    pts = np.array([[0.0], [0.0], [0.0], [10.0]])
    far_hits = 0
    trials = 200
    for s in range(trials):
        r = np.random.default_rng(s)
        c = kmcore._kmeans_pp_init(pts, 2, r)
        if c[1, 0] == 10.0:
            far_hits += 1
    assert far_hits / trials > 0.6


def test_kmeans_same_seed_reproducible():
    pts = _blobs()
    a = kmeans.KMeans(4, seed=42).fit(pts)
    b = kmeans.KMeans(4, seed=42).fit(pts)
    assert a.centers() == b.centers()
    assert a.error() == b.error()


def test_kmeans_different_seed_can_differ_and_best_is_picked():
    pts = _blobs()
    errors = {
        s: kmeans.KMeans(4, seed=s).fit(pts).error() for s in range(6)
    }
    # 同种子严格复现。
    for s in range(6):
        assert kmeans.KMeans(4, seed=s).fit(pts).error() == errors[s]


# ---------- 2. 收敛判据与中心==成员均值 ----------

def test_kmeans_centers_equal_member_means():
    pts = _blobs(300)
    m = kmeans.KMeans(4, seed=0).fit(pts)
    centers = np.asarray(m.centers())
    labels = np.argmin(kmcore._sq_dists(pts, centers), axis=1)
    for j in range(4):
        member_mean = pts[labels == j].mean(axis=0)
        np.testing.assert_allclose(
            centers[j], member_mean, rtol=0.0, atol=1e-12,
            err_msg=f"簇 {j} 中心与成员均值偏差超过 1e-12",
        )


def test_kmeans_respects_max_iter_cap():
    pts = _blobs(50)
    m = kmeans.KMeans(4, seed=0, max_iter=1).fit(pts)
    # 一轮迭代后仍返回 k 个中心且误差有限。
    assert len(m.centers()) == 4
    assert np.isfinite(m.error())
    # 即使达到迭代上限，收敛后中心仍必须等于成员均值（<=1e-12）。
    centers = np.asarray(m.centers())
    labels = np.argmin(kmcore._sq_dists(pts, centers), axis=1)
    for j in range(4):
        np.testing.assert_allclose(
            centers[j], pts[labels == j].mean(axis=0),
            rtol=0.0, atol=1e-12,
        )


def test_kmeans_tol_controls_convergence():
    pts = _blobs(200)
    tight = kmeans.KMeans(4, seed=0, tol=1e-13).fit(pts)
    loose = kmeans.KMeans(4, seed=0, tol=1e-2).fit(pts)
    # 两种容差都必须收敛到合法的成员均值不动点。
    for m in (tight, loose):
        c = np.asarray(m.centers())
        lab = np.argmin(kmcore._sq_dists(pts, c), axis=1)
        for j in range(4):
            np.testing.assert_allclose(
                c[j], pts[lab == j].mean(axis=0), rtol=0.0, atol=1e-12
            )


# ---------- 3. 空簇用最远点重新初始化 ----------

def test_kmeans_empty_cluster_reinit_keeps_k_clusters():
    # 只有 3 个紧凑点团却要求 6 个簇，必然反复制造空簇。
    pts = np.vstack([
        np.array([[0.0, 0.0, 0.0]]) + np.random.default_rng(i).normal(
            0.0, 1e-4, size=(30, 3))
        for i in range(3)
    ])
    m = kmeans.KMeans(6, seed=2).fit(pts)
    centers = np.asarray(m.centers())
    assert centers.shape == (6, 3)
    assert np.isfinite(centers).all()
    # 每个中心都必须实际拥有成员（精修后不存在死簇）。
    labels = np.argmin(kmcore._sq_dists(pts, centers), axis=1)
    assert set(np.unique(labels).tolist()) == set(range(6))


def test_kmeans_k_equal_n_each_point_is_center():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    m = kmeans.KMeans(3, seed=0).fit(pts)
    centers = sorted(map(tuple, np.asarray(m.centers())))
    assert centers == sorted(map(tuple, pts))
    assert m.error() == pytest.approx(0.0, abs=1e-12)


# ---------- 4. SSE 随 k 单调不增 ----------

def test_kmeans_error_monotonic_in_k_same_instance_chain():
    pts = _blobs(250)
    errors = [kmeans.KMeans(k, seed=3).fit(pts).error() for k in range(1, 9)]
    for lo, hi in zip(errors, errors[1:]):
        assert hi <= lo + 1e-9


def test_kmeans_error_monotonic_across_instances_and_unordered():
    pts = _blobs(250, seed=7)
    # 乱序拟合，跨实例同样必须单调不增。
    e5 = kmeans.KMeans(5, seed=11).fit(pts).error()
    e2 = kmeans.KMeans(2, seed=99).fit(pts).error()
    e7 = kmeans.KMeans(7, seed=0).fit(pts).error()
    assert e5 <= e2 + 1e-9
    assert e7 <= e5 + 1e-9


def test_kmeans_refit_does_not_increase_error():
    pts = _blobs(200)
    m = kmeans.KMeans(4, seed=5).fit(pts)
    first = m.error()
    m.fit(pts)
    assert m.error() <= first + 1e-12


# ---------- 5. 三次重启取最优 ----------

def test_kmeans_three_restarts_default():
    import inspect
    sig = inspect.signature(kmeans.KMeans.__init__)
    assert sig.parameters["n_init"].default == 3


def test_kmeans_more_restarts_not_worse():
    pts = _blobs(300, seed=4)
    e3 = kmeans.KMeans(4, seed=1, n_init=3).fit(pts).error()
    e10 = kmeans.KMeans(4, seed=1, n_init=10).fit(pts).error()
    assert e10 <= e3 + 1e-12


# ---------- 6. 性能 ----------

def test_kmeans_perf_10k_pixels_8_colors_under_300ms():
    # 在子进程中测量冷态单次拟合，排除测试进程内缓存/预热状态干扰，
    # 与题面"1 万像素 8 色不超过 300 ms"的冷启动口径一致。
    code = (
        "import time, numpy as np, kmeans;"
        "rng=np.random.default_rng(0);"
        "pts=rng.random((10000,3));"
        "t=time.perf_counter();"
        "m=kmeans.KMeans(8, seed=0).fit(pts);"
        "print((time.perf_counter()-t)*1000, len(m.centers()))"
    )
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
    )
    elapsed_ms, n_centers = proc.stdout.strip().split()
    assert int(n_centers) == 8
    assert float(elapsed_ms) < 300.0, (
        f"冷态耗时 {elapsed_ms} ms 超过 300 ms 上限"
    )


# ---------- 7. 可区分报错 ----------

def test_kmeans_invalid_k_errors():
    with pytest.raises(ValueError):
        kmeans.KMeans(0)
    with pytest.raises(ValueError):
        kmeans.KMeans(-3)
    with pytest.raises(ValueError):
        kmeans.KMeans(2.5)
    with pytest.raises(ValueError):
        kmeans.KMeans(True)
    with pytest.raises(ValueError):
        kmeans.KMeans(2, max_iter=0)
    with pytest.raises(ValueError):
        kmeans.KMeans(2, n_init=0)


def test_kmeans_k_larger_than_n_is_value_error():
    with pytest.raises(ValueError):
        kmeans.KMeans(5, seed=0).fit(np.zeros((3, 3)))


def test_kmeans_empty_input_distinct_error():
    with pytest.raises(EmptyInputError):
        kmeans.KMeans(2).fit([])
    with pytest.raises(EmptyInputError):
        kmeans.KMeans(2).fit(np.zeros((0, 3)))
    with pytest.raises(EmptyInputError):
        kmeans.KMeans(2).fit(None)


def test_kmeans_nonfinite_distinct_error():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 1.0]])
    with pytest.raises(NonFiniteColorError):
        kmeans.KMeans(2).fit(pts)
    pts2 = np.array([[0.0, 0.0, 0.0], [1.0, np.inf, 1.0]])
    with pytest.raises(NonFiniteColorError):
        kmeans.KMeans(2).fit(pts2)


def test_kmeans_error_types_are_distinguishable():
    # 三类错误必须能用 except 子句区分，而不是同一个异常类。
    assert issubclass(EmptyInputError, ValueError)
    assert issubclass(NonFiniteColorError, ValueError)
    assert EmptyInputError is not NonFiniteColorError
    try:
        kmeans.KMeans(2).fit([[np.nan, 0.0, 0.0], [1.0, 1.0, 1.0]])
    except EmptyInputError:
        pytest.fail("非有限输入被误判为空输入")
    except NonFiniteColorError:
        pass


# ---------- 8. 线程安全与接口 ----------

def test_kmeans_thread_safe_concurrent_fits():
    pts = _blobs(200, seed=9)
    results = []
    errors_out = []

    def worker(seed):
        try:
            m = kmeans.KMeans(4, seed=seed).fit(pts)
            results.append((seed, m.centers(), m.error()))
        except Exception as exc:  # pragma: no cover - 仅用于失败诊断
            errors_out.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors_out
    # 同种子结果在并发下仍与串行结果一致。
    for seed, centers, sse in results:
        ref = kmeans.KMeans(4, seed=seed).fit(pts)
        assert centers == ref.centers()
        assert sse == ref.error()


def test_kmeans_returns_list_of_tuples_and_float_error():
    m = kmeans.KMeans(3, seed=0).fit(_blobs(100))
    assert isinstance(m.centers(), list)
    assert all(isinstance(c, tuple) and len(c) == 3 for c in m.centers())
    assert isinstance(m.error(), float)
    assert m.error() >= 0.0


def test_kmeans_accessors_before_fit_raise():
    m = kmeans.KMeans(2)
    with pytest.raises(Exception):
        m.centers()
    with pytest.raises(Exception):
        m.error()


def test_kmeans_accepts_python_sequences():
    pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
           (0.5, 0.5, 0.5), (0.9, 0.9, 0.9)]
    m = kmeans.KMeans(2, seed=0).fit(pts)
    assert len(m.centers()) == 2
    assert m.error() >= 0.0
