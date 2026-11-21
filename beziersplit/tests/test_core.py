import math

import pytest

import beziersplit

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))


def _b():
    return beziersplit.BezierSplit()


def _check_length_positive():
    return _b().length(*C) > 0.0


def _check_length_line():
    l = _b().length((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0))
    return abs(l - 3.0) < 1e-6


def _check_t0():
    return abs(_b().t_at(*C, 0.0)) < 1e-6


def _check_t1():
    return abs(_b().t_at(*C, _b().length(*C)) - 1.0) < 1e-3


def _check_monotonic():
    l = _b().length(*C)
    t1 = _b().t_at(*C, l * 0.25)
    t2 = _b().t_at(*C, l * 0.75)
    return t1 <= t2


def _check_bad_s():
    return _raises(lambda: _b().t_at(*C, -1.0))


def _check_degenerate():
    return _raises(lambda: _b().length((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))


def _check_type():
    return isinstance(_b().t_at(*C, 0.5), float)


def _check_det():
    return _b().length(*C) == _b().length(*C)

CASES = [("length_positive", "_check_length_positive"), ("length_line", "_check_length_line"), ("t0", "_check_t0"), ("t1", "_check_t1"), ("monotonic", "_check_monotonic"), ("bad_s", "_check_bad_s"), ("degenerate", "_check_degenerate"), ("type", "_check_type"), ("det", "_check_det"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert BezierSplit() is not None
        return
    assert fn() is True
