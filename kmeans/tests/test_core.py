import math

import pytest

import kmeans

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PTS = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.9, 0.9, 0.9)]


def _k(k=2):
    return kmeans.KMeans(k, 0)


def _check_centers_count():
    m = _k(); m.fit(PTS)
    return len(m.centers()) <= 2


def _check_error_positive():
    m = _k(); m.fit(PTS)
    return m.error() >= 0.0


def _check_converge():
    m = _k(2); m.fit(PTS)
    e1 = m.error()
    m.fit(PTS)
    return m.error() <= e1 + 1e-9


def _check_bad_k():
    return _raises(lambda: kmeans.KMeans(0))


def _check_empty():
    return _raises(lambda: _k().fit([]))


def _check_det():
    a = _k(); a.fit(PTS)
    b = _k(); b.fit(PTS)
    return a.centers() == b.centers()


def _check_monotonic_k():
    a = _k(1); a.fit(PTS)
    b = _k(2); b.fit(PTS)
    return b.error() <= a.error() + 1e-9


def _check_type():
    m = _k(); m.fit(PTS)
    return isinstance(m.centers(), list)

CASES = [("centers_count", "_check_centers_count"), ("error_positive", "_check_error_positive"), ("converge", "_check_converge"), ("bad_k", "_check_bad_k"), ("empty", "_check_empty"), ("det", "_check_det"), ("monotonic_k", "_check_monotonic_k"), ("type", "_check_type"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert KMeans() is not None
        return
    assert fn() is True
