"""DFT 完整规格测试（用例名均含 dft，可被 `pytest -k dft` 选中）。"""
import cmath
import math
import threading
import time

import pytest

import dft

TWO_PI = 2.0 * math.pi


def _naive_dft(x):
    """朴素 DFT 定义，作为一致性基准。"""
    n = len(x)
    return [
        sum(complex(v) * cmath.exp(-1j * TWO_PI * k * j / n) for j, v in enumerate(x))
        for k in range(n)
    ]


@pytest.fixture
def d():
    return dft.Dft()


def test_dft_matches_naive_definition(d):
    # 2 的幂、奇数、素数、混合因子长度
    for n in [1, 2, 3, 4, 5, 7, 8, 11, 16, 17, 31, 64, 100, 127, 251, 256, 300]:
        x = [math.sin(0.7 * j) + 0.5 * math.cos(0.31 * j + 1.0) for j in range(n)]
        out = d.transform(x)
        assert isinstance(out, list)
        assert len(out) == n
        err = max(abs(a - b) for a, b in zip(out, _naive_dft(x)))
        assert err <= 1e-9, f"n={n} 与朴素 DFT 最大误差 {err}"


def test_dft_complex_input(d):
    x = [complex(math.cos(0.2 * j), math.sin(0.11 * j)) for j in range(101)]
    err = max(abs(a - b) for a, b in zip(d.transform(x), _naive_dft(x)))
    assert err <= 1e-9


def test_dft_dc_and_impulse(d):
    out = d.transform([1.0, 0.0, 0.0, 0.0])
    assert abs(abs(out[0]) - 1.0) < 1e-9


def test_dft_parseval_energy(d):
    for n in [4, 10, 100, 997]:
        x = [math.sin(j * 0.5) + 1.0 for j in range(n)]
        e_time = sum(abs(v) ** 2 for v in x)
        e_freq = sum(abs(v) ** 2 for v in d.transform(x)) / n
        assert abs(e_time - e_freq) <= 1e-9 * max(1.0, e_time)


def test_dft_deterministic(d):
    s = [1.0, 0.0, 0.0, 0.0]
    assert d.transform(s) == d.transform(s)


def test_dft_non_power_of_two_lengths(d):
    for n in [3, 6, 9, 10, 12, 15, 100, 1001]:
        out = d.transform([float(j) for j in range(n)])
        assert len(out) == n
        # DC 分量等于时域求和，与长度是否为 2 的幂无关
        assert abs(out[0] - sum(range(n))) < 1e-6


def test_dft_invalid_length_raises(d):
    with pytest.raises(ValueError):
        d.transform([])


def test_dft_non_finite_input_classified(d):
    with pytest.raises(ValueError):
        d.transform([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError):
        d.transform([1.0, float("inf")])
    with pytest.raises(ValueError):
        d.transform([complex(0.0, float("-inf"))])


def test_dft_bad_element_type(d):
    with pytest.raises(TypeError):
        d.transform([1.0, "x"])
    with pytest.raises(TypeError):
        d.transform("abc")
    with pytest.raises(TypeError):
        d.transform(123)


def test_dft_10000_points_under_two_seconds(d):
    x = [math.sin(0.01 * j) + 0.3 * math.cos(0.003 * j) for j in range(10000)]
    t0 = time.perf_counter()
    out = d.transform(x)
    elapsed = time.perf_counter() - t0
    assert len(out) == 10000
    assert elapsed < 2.0, f"耗时 {elapsed:.3f}s 超过 2s 上界"
    # 大长度下用 DC + Parseval 间接校验正确性
    assert abs(out[0] - sum(x)) < 1e-7
    e_freq = sum(abs(v) ** 2 for v in out) / 10000
    e_time = sum(abs(v) ** 2 for v in x)
    assert abs(e_time - e_freq) / e_time < 1e-9


def test_dft_thread_safe_concurrent(d):
    x = [math.sin(0.01 * j) for j in range(4096)]
    results = [None] * 8
    errors = []

    def worker(i):
        try:
            results[i] = d.transform(x)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(r == results[0] for r in results)


def test_dft_window_length_and_endpoints(d):
    for n in [1, 2, 3, 8, 100, 10000]:
        w = d.window(n)
        assert isinstance(w, list)
        assert len(w) == n
        assert all(isinstance(v, float) for v in w)
        if n >= 2:
            assert w[0] == 0.0 and w[-1] == 0.0
            ref = [0.5 * (1.0 - math.cos(TWO_PI * k / (n - 1))) for k in range(n)]
            assert max(abs(a - b) for a, b in zip(w, ref)) <= 1e-12
        else:
            assert w == [1.0]


def test_dft_window_hamming(d):
    w = d.window(8, "hamming")
    assert len(w) == 8
    assert abs(w[0] - 0.08) < 1e-12
    assert abs(w[-1] - 0.08) < 1e-12


def test_dft_window_invalid_length_raises(d):
    with pytest.raises(ValueError):
        d.window(0)
    with pytest.raises(ValueError):
        d.window(-3)
    with pytest.raises(TypeError):
        d.window(8.0)
    with pytest.raises(TypeError):
        d.window(True)
    with pytest.raises(ValueError):
        d.window(8, "rect")
