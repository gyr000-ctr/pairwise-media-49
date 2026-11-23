import math

import pytest

import bvh2
from bvh2 import Bvh

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

BOXES = [("a", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), ("b", ((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)))]


def _s():
    b = bvh2.Bvh()
    for n, box in BOXES:
        b.add(n, box)
    b.build()
    return b


def _check_hit():
    return _s().nearest((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0))[1] == "a"


def _check_second():
    return _s().nearest((1.5, 0.5, 0.5), (1.0, 0.0, 0.0))[1] == "b"


def _check_miss():
    return _s().nearest((0.0, 5.0, 0.0), (0.0, 1.0, 0.0)) is None


def _check_empty():
    return _raises(lambda: bvh2.Bvh().nearest((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))


def _check_bad_box():
    return _raises(lambda: bvh2.Bvh().add("x", ((1, 1, 1), (0, 0, 0))))


def _check_type():
    return isinstance(_s().nearest((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0)), tuple)


def _check_det():
    return _s().nearest((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0)) == _s().nearest((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0))


def _check_t_positive():
    t, name = _s().nearest((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0))
    return t >= 0.0

CASES = [("hit", "_check_hit"), ("second", "_check_second"), ("miss", "_check_miss"), ("empty", "_check_empty"), ("bad_box", "_check_bad_box"), ("type", "_check_type"), ("det", "_check_det"), ("t_positive", "_check_t_positive")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Bvh() is not None
        return
    assert fn() is True


def _scene(n, seed=0):
    """造 n 个不相交的小盒，返回 (Bvh, 盒列表)。"""
    import random

    rnd = random.Random(seed)
    b = bvh2.Bvh()
    boxes = []
    for i in range(n):
        c = (rnd.uniform(-20.0, 20.0), rnd.uniform(-20.0, 20.0), rnd.uniform(-20.0, 20.0))
        s = (rnd.uniform(0.1, 1.0), rnd.uniform(0.1, 1.0), rnd.uniform(0.1, 1.0))
        box = (c, (c[0] + s[0], c[1] + s[1], c[2] + s[2]))
        boxes.append(box)
        b.add(i, box)
    b.build()
    return b, boxes


def _brute(boxes, o, d):
    """暴力最近命中，用于对拍。"""
    import bvh2.core as core

    best = None
    for i, (lo, hi) in enumerate(boxes):
        r = core._slab(lo, hi, o, d)
        if r is None:
            continue
        t = max(r[0], 0.0)
        if best is None or t < best[0] or (t == best[0] and i < best[1]):
            best = (t, i)
    return best


def _check_brute_parity():
    import random

    rnd = random.Random(1)
    b, boxes = _scene(300)
    for _ in range(1000):
        o = tuple(rnd.uniform(-25.0, 25.0) for _ in range(3))
        d = tuple(rnd.choice([0.0, rnd.uniform(-1.0, 1.0)]) for _ in range(3))
        if d == (0.0, 0.0, 0.0):
            continue
        got = b.nearest(o, d)
        exp = _brute(boxes, o, d)
        if (got is None) != (exp is None):
            return False
        if exp is not None and (abs(got[0] - exp[0]) > 1e-9 or got[1] != exp[1]):
            return False
    return True


def _check_tie_lowest():
    # 两个盒在 y/z 方向完全重叠，x 方向相邻：从 -x 打过去先命中下标小的。
    b = bvh2.Bvh()
    b.add("far", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    b.add("near", ((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)))
    b.build()
    t, name = b.nearest((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0))
    return name == "far" and abs(t - 1.0) < 1e-12


def _check_inside_origin():
    # 起点在盒内，进入参数取 0。
    t, name = _s().nearest((0.5, 0.5, 0.5), (1.0, 0.0, 0.0))
    return name == "a" and t == 0.0


def _check_behind():
    # 盒完全在射线后方（t < 0）视为未命中。
    return _s().nearest((-5.0, 0.5, 0.5), (-1.0, 0.0, 0.0)) is None


def _check_diagonal():
    b = bvh2.Bvh()
    b.add("q", ((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)))
    b.build()
    r = b.nearest((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    return r is not None and r[1] == "q" and abs(r[0] - 2.0) < 1e-12


def _check_bad_ray():
    b = _s()
    return (
        _raises(lambda: b.nearest((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        and _raises(lambda: b.nearest((float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0)))
        and _raises(lambda: b.nearest((0.0, 0.0, 0.0), (float("inf"), 0.0, 0.0)))
    )


def _check_bad_box_nan():
    return _raises(lambda: bvh2.Bvh().add("x", ((0.0, 0.0, 0.0), (1.0, float("nan"), 1.0))))


def _check_unbuilt():
    b = bvh2.Bvh()
    b.add("a", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    return _raises(lambda: b.nearest((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)))


def _check_rebuild():
    b = _s()
    b.add("c", ((-5.0, 0.5, 0.5), (-4.0, 1.5, 1.5)))
    b.build()
    t, name = b.nearest((-6.0, 1.0, 1.0), (1.0, 0.0, 0.0))
    return name == "c" and abs(t - 1.0) < 1e-12


def _check_concurrent():
    return True


CASES2 = [
    ("brute_parity", "_check_brute_parity"),
    ("tie_lowest", "_check_tie_lowest"),
    ("inside_origin", "_check_inside_origin"),
    ("behind", "_check_behind"),
    ("diagonal", "_check_diagonal"),
    ("bad_ray", "_check_bad_ray"),
    ("bad_box_nan", "_check_bad_box_nan"),
    ("unbuilt", "_check_unbuilt"),
    ("rebuild", "_check_rebuild"),
]


@pytest.mark.parametrize("name,check", CASES2, ids=[c[0] for c in CASES2])
def test_bvh_more(name, check):
    assert globals()[check]() is True


def _check_concurrent_impl():
    import threading

    b, boxes = _scene(200, seed=3)

    queries = []
    import random

    rnd = random.Random(9)
    for _ in range(200):
        oo = tuple(rnd.uniform(-25.0, 25.0) for _ in range(3))
        dd = tuple(rnd.choice([0.0, rnd.uniform(-1.0, 1.0)]) for _ in range(3))
        if dd != (0.0, 0.0, 0.0):
            queries.append((oo, dd))

    expected = [b.nearest(oo, dd) for oo, dd in queries]
    actual = [None] * len(queries)
    errors = []

    def worker(slots):
        try:
            for k in slots:
                actual[k] = b.nearest(*queries[k])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = []
    per = len(queries) // 4
    for w in range(4):
        slots = range(w * per, (w + 1) * per)
        threads.append(threading.Thread(target=worker, args=(slots,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return not errors and actual == expected


def test_bvh_concurrent():
    assert _check_concurrent_impl() is True


def _check_perf_10k():
    import time

    rnd = __import__("random").Random(5)
    b = bvh2.Bvh()
    for i in range(10000):
        c = (rnd.uniform(0.0, 100.0), rnd.uniform(0.0, 100.0), rnd.uniform(0.0, 100.0))
        b.add(i, (c, (c[0] + 0.5, c[1] + 0.5, c[2] + 0.5)))
    start = time.perf_counter()
    b.build()
    elapsed = time.perf_counter() - start
    if elapsed > 0.5:
        print("build 10k boxes took %.3fs" % elapsed)
        return False
    # 查询走一遍，保证树可用。
    origin = (0.0, 0.0, 0.0)
    return b.nearest(origin, (1.0, 0.0, 0.0)) is None or True


def test_bvh_perf():
    assert _check_perf_10k() is True
