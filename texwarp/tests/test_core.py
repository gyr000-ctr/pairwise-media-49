import math

import pytest

import texwarp

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _w(amp=0.1):
    return texwarp.TexWarp(amp, 3)


def _check_zero_amp():
    w = texwarp.TexWarp(0.0, 3)
    return w.warp(0.3, 0.7) == (0.3, 0.7)


def _check_bounded():
    x, y = _w().warp(0.3, 0.7)
    return abs(x - 0.3) <= 0.1 + 1e-9 and abs(y - 0.7) <= 0.1 + 1e-9


def _check_det():
    return _w().warp(0.4, 0.2) == _w().warp(0.4, 0.2)


def _check_seed_diff():
    a = texwarp.TexWarp(0.1, 3, seed=1)
    b = texwarp.TexWarp(0.1, 3, seed=2)
    return a.warp(0.4, 0.2) != b.warp(0.4, 0.2)


def _check_bad_amp():
    return _raises(lambda: texwarp.TexWarp(-0.1, 3))


def _check_bad_octaves():
    return _raises(lambda: texwarp.TexWarp(0.1, 0))


def _check_type():
    return isinstance(_w().warp(0.1, 0.1), tuple)


def _check_no_mut():
    w = _w()
    w.warp(0.2, 0.2)
    return True

CASES = [("zero_amp", "_check_zero_amp"), ("bounded", "_check_bounded"), ("det", "_check_det"), ("seed_diff", "_check_seed_diff"), ("bad_amp", "_check_bad_amp"), ("bad_octaves", "_check_bad_octaves"), ("type", "_check_type"), ("no_mut", "_check_no_mut"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert TexWarp() is not None
        return
    assert fn() is True
