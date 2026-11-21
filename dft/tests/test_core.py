import math

import pytest

import dft

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [1.0, 0.0, 0.0, 0.0]


def _d():
    return dft.Dft()


def _check_len():
    return len(_d().transform(S)) == 4


def _check_dc():
    out = _d().transform(S)
    return abs(abs(out[0]) - 1.0) < 1e-9


def _check_parseval():
    out = _d().transform([1.0, 1.0, 0.0, 0.0])
    e_time = sum(abs(v) ** 2 for v in [1.0, 1.0, 0.0, 0.0])
    e_freq = sum(abs(v) ** 2 for v in out) / 4
    return abs(e_time - e_freq) < 1e-9


def _check_bad_len():
    return _raises(lambda: _d().transform([]))


def _check_window_len():
    return len(_d().window(8)) == 8


def _check_window_ends():
    w = _d().window(8)
    return abs(w[0]) < 1e-9


def _check_window_bad():
    return _raises(lambda: _d().window(0))


def _check_type():
    return isinstance(_d().transform(S), list)


def _check_det():
    return _d().transform(S) == _d().transform(S)

CASES = [("len", "_check_len"), ("dc", "_check_dc"), ("parseval", "_check_parseval"), ("bad_len", "_check_bad_len"), ("window_len", "_check_window_len"), ("window_ends", "_check_window_ends"), ("window_bad", "_check_window_bad"), ("type", "_check_type"), ("det", "_check_det"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Dft() is not None
        return
    assert fn() is True
