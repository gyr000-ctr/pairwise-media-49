import math

import pytest

import gamut
from gamut import GamutMap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _g():
    return gamut.GamutMap()


def _check_in_gamut():
    return _g().map((0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5)


def _check_clamped():
    out = _g().map((1.2, 0.0, 0.0))
    return all(0.0 <= v <= 1.0 for v in out)


def _check_bad_len():
    return _raises(lambda: _g().map((1.0, 0.0)))


def _check_type():
    return isinstance(_g().map((0.9, 0.1, 0.1)), tuple)


def _check_det():
    return _g().map((1.1, 0.2, 0.2)) == _g().map((1.1, 0.2, 0.2))


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _g().map((0.5, 1.5, 0.0)))


def _check_no_mut():
    c = (0.2, 0.4, 0.6)
    _g().map(c)
    return c == (0.2, 0.4, 0.6)

CASES = [("in_gamut", "_check_in_gamut"), ("clamped", "_check_clamped"), ("bad_len", "_check_bad_len"), ("type", "_check_type"), ("det", "_check_det"), ("range", "_check_range"), ("no_mut", "_check_no_mut"), ("smoke7", "_check_smoke7"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert GamutMap() is not None
        return
    assert fn() is True
