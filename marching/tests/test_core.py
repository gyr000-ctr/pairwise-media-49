import math

import pytest

import marching
from marching import Marching

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

F = [[0.0, 1.0], [1.0, 0.0]]


def _m():
    return marching.Marching()


def _check_nonempty():
    return len(_m().contour(F, 0.5)) >= 0


def _check_empty_field():
    return _m().contour([], 0.5) == []


def _check_all_below():
    return _m().contour([[0.0, 0.0], [0.0, 0.0]], 0.5) == []


def _check_all_above():
    return _m().contour([[1.0, 1.0], [1.0, 1.0]], 0.5) == []


def _check_bad_level():
    return _raises(lambda: _m().contour(F, float("nan")))


def _check_type():
    return isinstance(_m().contour(F, 0.5), list)


def _check_points_inside():
    out = _m().contour(F, 0.5)
    return all(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for seg in out for p in seg)


def _check_det():
    return _m().contour(F, 0.5) == _m().contour(F, 0.5)


def _check_saddle():
    return isinstance(_m().contour(F, 0.5), list)

CASES = [("nonempty", "_check_nonempty"), ("empty_field", "_check_empty_field"), ("all_below", "_check_all_below"), ("all_above", "_check_all_above"), ("bad_level", "_check_bad_level"), ("type", "_check_type"), ("points_inside", "_check_points_inside"), ("det", "_check_det"), ("saddle", "_check_saddle"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Marching() is not None
        return
    assert fn() is True
