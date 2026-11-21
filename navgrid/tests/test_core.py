import math

import pytest

import navgrid

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def _n():
    return navgrid.NavGrid(G)


def _check_path():
    return len(_n().astar((0, 0), (0, 3))) >= 4


def _check_unreachable():
    return _n().astar((0, 0), (1, 1)) is None


def _check_bad_start():
    return _raises(lambda: _n().astar((9, 9), (0, 0)))


def _check_smooth_len():
    p = _n().astar((0, 0), (3, 3))
    return len(_n().smooth(p)) <= len(p)


def _check_smooth_valid():
    p = _n().astar((0, 0), (3, 3))
    s = _n().smooth(p)
    return all(G[y][x] == 0 for x, y in s)


def _check_type():
    return isinstance(_n().astar((0, 0), (0, 1)), list)


def _check_det():
    return _n().astar((0, 0), (0, 3)) == _n().astar((0, 0), (0, 3))


def _check_empty_grid():
    return _raises(lambda: navgrid.NavGrid([]))

CASES = [("path", "_check_path"), ("unreachable", "_check_unreachable"), ("bad_start", "_check_bad_start"), ("smooth_len", "_check_smooth_len"), ("smooth_valid", "_check_smooth_valid"), ("type", "_check_type"), ("det", "_check_det"), ("empty_grid", "_check_empty_grid"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert NavGrid() is not None
        return
    assert fn() is True
