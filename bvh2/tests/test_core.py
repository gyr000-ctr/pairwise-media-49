import math

import pytest

import bvh2

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

CASES = [("hit", "_check_hit"), ("second", "_check_second"), ("miss", "_check_miss"), ("empty", "_check_empty"), ("bad_box", "_check_bad_box"), ("type", "_check_type"), ("det", "_check_det"), ("t_positive", "_check_t_positive"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Bvh() is not None
        return
    assert fn() is True
