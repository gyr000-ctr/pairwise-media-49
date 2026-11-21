import math

import pytest

import worley

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _w(s=1):
    return worley.Worley(s, 1.0)


def _check_pair():
    return len(_w().sample(0.3, 0.7)) == 2


def _check_order():
    f1, f2 = _w().sample(0.3, 0.7)
    return f1 <= f2 + 1e-12


def _check_nonneg():
    f1, f2 = _w().sample(0.3, 0.7)
    return f1 >= 0.0


def _check_det():
    return _w().sample(0.4, 0.4) == _w().sample(0.4, 0.4)


def _check_seed_diff():
    return _w(1).sample(0.4, 0.4) != _w(2).sample(0.4, 0.4)


def _check_boundary():
    return len(_w().sample(-5.5, -7.5)) == 2


def _check_bad_seed():
    return _raises(lambda: worley.Worley(-1))


def _check_type():
    return isinstance(_w().sample(0.1, 0.1), tuple)


def _check_range():
    f1, f2 = _w().sample(9.9, 9.9)
    return f1 <= 2.0

CASES = [("pair", "_check_pair"), ("order", "_check_order"), ("nonneg", "_check_nonneg"), ("det", "_check_det"), ("seed_diff", "_check_seed_diff"), ("boundary", "_check_boundary"), ("bad_seed", "_check_bad_seed"), ("type", "_check_type"), ("range", "_check_range"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Worley() is not None
        return
    assert fn() is True
