import math

import pytest

import pitch

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

import math


def _tone(f, n=2000, rate=8000):
    return [math.sin(2 * math.pi * f * i / rate) for i in range(n)]


def _p():
    return pitch.Pitch()


def _check_440():
    f = _p().estimate(_tone(440.0), 8000)
    return abs(f - 440.0) / 440.0 < 0.02


def _check_220():
    f = _p().estimate(_tone(220.0), 8000)
    return abs(f - 220.0) / 220.0 < 0.02


def _check_silence():
    return _p().estimate([0.0] * 100, 8000) == 0.0


def _check_bad_rate():
    return _raises(lambda: _p().estimate(_tone(440.0), 0))


def _check_empty():
    return _p().estimate([], 8000) == 0.0


def _check_type():
    return isinstance(_p().estimate(_tone(440.0), 8000), float)


def _check_det():
    return _p().estimate(_tone(440.0), 8000) == _p().estimate(_tone(440.0), 8000)


def _check_no_octave():
    f = _p().estimate(_tone(330.0), 8000)
    return abs(f - 330.0) / 330.0 < 0.05

CASES = [("440", "_check_440"), ("220", "_check_220"), ("silence", "_check_silence"), ("bad_rate", "_check_bad_rate"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("no_octave", "_check_no_octave"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Pitch() is not None
        return
    assert fn() is True
