"""pitch 验收测试：基频精度、倍频消歧、置信度、噪声、性能与错误分类。

``-k pitch`` 通过模块名匹配本文件，因此题面验收命令可直接收集这些用例。
"""

import math
import random
import threading
import time

import pytest

import pitch

RATE = 8000


def _tone(freq, n=2000, rate=RATE):
    return [math.sin(2.0 * math.pi * freq * i / rate) for i in range(n)]


def _noisy(freq, n=4000, rate=RATE, snr_db=5.0, seed=0):
    rng = random.Random(seed)
    noise_std = math.sqrt(0.5 * 10.0 ** (-snr_db / 10.0))
    return [s + rng.gauss(0.0, noise_std) for s in _tone(freq, n, rate)]


def _err(freq, reference):
    return abs(freq - reference) / reference


# --------------------------------------------------------------------------
# 基本精度
# --------------------------------------------------------------------------

PURE_FREQS = [80.0, 82.0, 110.0, 220.0, 330.0, 440.0, 523.251, 660.0, 880.0, 997.0, 1000.0]


@pytest.mark.parametrize("freq", PURE_FREQS)
def test_pitch_pure_tone_within_one_percent(freq):
    got = pitch.Pitch().estimate(_tone(freq), RATE)
    assert _err(got, freq) <= 0.01


@pytest.mark.parametrize("rate", [8000, 11025, 16000, 22050, 44100, 48000])
def test_pitch_various_sample_rates(rate):
    for freq in (100.0, 440.0, 900.0):
        got = pitch.Pitch().estimate(_tone(freq, 4000, rate), rate)
        assert _err(got, freq) <= 0.01


def test_pitch_returns_float_and_confidence():
    result = pitch.Pitch().estimate(_tone(440.0), RATE)
    assert isinstance(result, float)
    assert 0.0 <= result.confidence <= 1.0


# --------------------------------------------------------------------------
# 倍频 / 半频歧义
# --------------------------------------------------------------------------

@pytest.mark.parametrize("freq", [110.0, 220.0, 330.0, 440.0])
def test_pitch_no_octave_error(freq):
    got = pitch.Pitch().estimate(_tone(freq), RATE)
    assert _err(got, freq) <= 0.01, "倍频/半频误判: %r" % (got,)


@pytest.mark.parametrize("missing", [(2, 3, 4, 5), (3, 5, 7), (2, 4, 6, 8)])
def test_pitch_missing_fundamental_picks_lowest_common(missing):
    """基频缺失时，应选能被所有谐波整除的最低频率。

    当所有谐波的公共因子 > 1（如只有 2x/4x/6x/8x）时，信号真正的
    可观测基频是 f0*gcd，而不是再往下硬凑一个不存在的次谐波。
    """
    f0 = 150.0
    signal = [
        sum(math.sin(2.0 * math.pi * m * f0 * i / RATE) / m for m in missing)
        for i in range(4000)
    ]
    common = 0
    for m in missing:
        common = math.gcd(common, m)
    expected = f0 * common
    got = pitch.Pitch().estimate(signal, RATE)
    assert _err(got, expected) <= 0.01


def test_pitch_harmonic_rich_prefers_fundamental():
    f0 = 200.0
    signal = [
        sum(math.sin(2.0 * math.pi * m * f0 * i / RATE) / m for m in range(1, 9))
        for i in range(4000)
    ]
    got = pitch.Pitch().estimate(signal, RATE)
    assert _err(got, f0) <= 0.01


# --------------------------------------------------------------------------
# 置信度
# --------------------------------------------------------------------------

def test_pitch_silence_confidence_zero():
    result = pitch.Pitch().estimate([0.0] * 500, RATE)
    assert float(result) == 0.0
    assert result.confidence == 0.0


def test_pitch_empty_is_silence():
    result = pitch.Pitch().estimate([], RATE)
    assert float(result) == 0.0
    assert result.confidence == 0.0


def test_pitch_noise_only_confidence_zero():
    rng = random.Random(7)
    result = pitch.Pitch().estimate([rng.gauss(0.0, 1.0) for _ in range(4000)], RATE)
    assert result.confidence == 0.0


NON_TONAL = {
    "ramp": [float(i) for i in range(2000)],
    "quadratic": [float(i * i) for i in range(2000)],
    "impulse_train": [1.0 if i % 50 == 0 else 0.0 for i in range(2000)],
    "two_tones": [
        math.sin(2.0 * math.pi * 300 * i / RATE)
        + math.sin(2.0 * math.pi * 611 * i / RATE)
        for i in range(4000)
    ],
}


@pytest.mark.parametrize("name", sorted(NON_TONAL))
def test_pitch_non_tonal_confidence_zero(name):
    """非纯音（斜坡/冲击串/双音）不得给出正置信度。"""
    result = pitch.Pitch().estimate(NON_TONAL[name], RATE)
    assert result.confidence == 0.0


@pytest.mark.parametrize("freq", [80.0, 220.0, 440.0, 1000.0])
def test_pitch_pure_tone_confidence_high(freq):
    result = pitch.Pitch().estimate(_tone(freq), RATE)
    assert result.confidence >= 0.9


# --------------------------------------------------------------------------
# 噪声
# --------------------------------------------------------------------------

@pytest.mark.parametrize("freq", [100.0, 220.0, 330.0, 440.0, 660.0, 880.0])
def test_pitch_five_db_noise_within_three_percent(freq):
    got = pitch.Pitch().estimate(_noisy(freq, seed=int(freq)), RATE)
    assert _err(got, freq) <= 0.03


# --------------------------------------------------------------------------
# 性能
# --------------------------------------------------------------------------

def test_pitch_ten_thousand_samples_under_300ms():
    samples = _tone(220.0, 10000, 44100)
    estimator = pitch.Pitch()
    estimator.estimate(samples, 44100)  # 预热
    start = time.perf_counter()
    estimator.estimate(samples, 44100)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.300, "耗时 %.1f ms" % (elapsed * 1000.0)


# --------------------------------------------------------------------------
# 错误分类
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad_rate", [0, -1, -8000.0, float("nan"), float("inf")])
def test_pitch_bad_rate_raises_distinct_error(bad_rate):
    with pytest.raises(pitch.InvalidSampleRate):
        pitch.Pitch().estimate(_tone(440.0), bad_rate)


def test_pitch_bad_rate_not_caught_as_samples_error():
    with pytest.raises(pitch.InvalidSampleRate) as info:
        pitch.Pitch().estimate(_tone(440.0), 0)
    assert not isinstance(info.value, pitch.InvalidSamples)


@pytest.mark.parametrize("bad_rate", [None, "fast", object()])
def test_pitch_non_numeric_rate_raises(bad_rate):
    with pytest.raises(ValueError):
        pitch.Pitch().estimate(_tone(440.0), bad_rate)


@pytest.mark.parametrize("nan", [float("nan"), float("inf"), float("-inf")])
def test_pitch_non_finite_samples_raise(nan):
    samples = _tone(440.0)
    samples[100] = nan
    with pytest.raises(pitch.NonFiniteInput):
        pitch.Pitch().estimate(samples, RATE)


def test_pitch_non_finite_is_both_value_and_type_error():
    with pytest.raises(ValueError):
        pitch.Pitch().estimate([float("nan")] * 10, RATE)
    with pytest.raises(TypeError):
        pitch.Pitch().estimate([float("nan")] * 10, RATE)


def test_pitch_bad_samples_type_raises():
    with pytest.raises(pitch.InvalidSamples):
        pitch.Pitch().estimate("not samples", RATE)


def test_pitch_complex_samples_raise():
    import numpy as np

    with pytest.raises(pitch.InvalidSamples):
        pitch.Pitch().estimate(np.arange(64, dtype=complex), RATE)


# --------------------------------------------------------------------------
# 并发安全与确定性
# --------------------------------------------------------------------------

def test_pitch_deterministic():
    samples = _tone(330.0)
    a = pitch.Pitch().estimate(samples, RATE)
    b = pitch.Pitch().estimate(samples, RATE)
    assert float(a) == float(b)
    assert a.confidence == b.confidence


def test_pitch_concurrent_different_instances():
    cases = [(110.0, 0), (220.0, 1), (440.0, 2), (880.0, 3)] * 8
    results = [None] * len(cases)

    def worker(index, freq, seed):
        results[index] = pitch.Pitch().estimate(_noisy(freq, seed=seed), RATE)

    threads = [
        threading.Thread(target=worker, args=(i, f, s))
        for i, (f, s) in enumerate(cases)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for (freq, _seed), got in zip(cases, results):
        assert got is not None
        assert _err(got, freq) <= 0.03


def test_pitch_concurrent_shared_instance():
    estimator = pitch.Pitch()
    samples = {f: _tone(f, 3000) for f in (150.0, 300.0, 600.0)}
    results = {}
    lock = threading.Lock()

    def worker(freq):
        value = estimator.estimate(samples[freq], RATE)
        with lock:
            results[freq] = value

    threads = [threading.Thread(target=worker, args=(f,)) for f in samples for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for freq, got in results.items():
        assert _err(got, freq) <= 0.01
