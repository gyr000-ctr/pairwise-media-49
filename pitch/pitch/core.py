"""自相关基频估计 + 谐波校正。

算法概览
--------
1. 预处理：去均值、按峰值归一化；静音/空输入直接判为无基频。
2. YIN（累积均值归一化差分函数）：给出低八度错误较少的基频候选。
3. 频谱：Hann 窗补零 FFT，取显著谱峰。
4. 倍频/半频消歧：对候选基频（YIN 的 1/4、1/2、1x、2x，以及各谱峰除以 1..6）
   用“谱峰可解释度”打分，选能解释最多谱峰、且预期谐波缺失最少的候选。
5. 精修：对基频峰做 Hann 加窗 DTFT 单峰极大化（FFT 粗定位 + 黄金分割），
   谐波仅在基频缺失时用加权最小二乘。DTFT 对纯音的频率估计是渐近无偏的，
   即使在低采样率、短窗口下也能满足 1% 的频率误差。
6. 置信度：谐波解释度与基频周期处归一化自相关的较小者，落在 [0, 1]。

线程安全：模块内无可变全局状态，所有工作数组都是调用内局部变量。
"""

import math

import numpy as np

FMIN = 80.0
FMAX = 1000.0

__all__ = [
    "Pitch",
    "PitchResult",
    "PitchError",
    "InvalidInput",
    "InvalidSamples",
    "NonFiniteInput",
    "InvalidSampleRate",
]


class PitchResult(float):
    """基频估计结果。

    为了兼容既有 API（``estimate`` 返回浮点频率），本类是 ``float`` 的子类，
    数值等于基频 Hz；同时携带 ``confidence`` 并可解包为 ``(freq, conf)``::

        f, c = Pitch().estimate(samples, rate)
        f = Pitch().estimate(samples, rate)      # 仍是 float，可直接比较/运算
    """

    __slots__ = ("confidence",)

    def __new__(cls, frequency, confidence):
        obj = float.__new__(cls, frequency)
        obj.confidence = float(confidence)
        return obj

    @property
    def frequency(self):
        return float(self)

    def __iter__(self):
        yield float(self)
        yield self.confidence

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index in (0, -2):
            return float(self)
        if index in (1, -1):
            return self.confidence
        raise IndexError("PitchResult 只有 (freq, confidence) 两个分量")

    def __reduce__(self):
        return (self.__class__, (float(self), self.confidence))

    def __repr__(self):
        return "PitchResult(frequency=%r, confidence=%r)" % (
            float(self),
            self.confidence,
        )


class PitchError(ValueError):
    """pitch 输入错误基类（继承 ValueError，兼容既有调用约定）。"""


class InvalidInput(PitchError):
    """非法输入。"""


class InvalidSamples(InvalidInput, TypeError):
    """样本序列类型/形状非法（同时是 TypeError，可区分于采样率错误）。"""


class NonFiniteInput(InvalidSamples):
    """样本包含 NaN 或 Inf。"""


class InvalidSampleRate(InvalidInput):
    """采样率非法（非数字、非有限或 <= 0）。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _prepare(samples):
    """校验并归一化输入；返回 1-D float64 数组，静音返回 None。"""
    if isinstance(samples, (bytes, bytearray, str)):
        raise InvalidSamples("samples 必须是实数序列，不能是文本/字节")
    if isinstance(samples, np.ndarray) and np.iscomplexobj(samples):
        raise InvalidSamples("samples 必须是实数序列，不能是复数")
    try:
        arr = np.asarray(samples, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidSamples("samples 必须是可转换为浮点数的序列") from exc
    if arr.ndim == 0:
        raise InvalidSamples("samples 必须是一维序列")
    if arr.ndim != 1:
        arr = arr.reshape(-1) if arr.size and min(arr.shape) == 1 else arr
        if arr.ndim != 1:
            raise InvalidSamples("samples 必须是一维序列")
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        raise NonFiniteInput("samples 包含 NaN 或 Inf")
    arr = arr - arr.mean()
    peak = float(np.max(np.abs(arr)))
    if peak == 0.0:
        return None
    return arr / peak


def _check_rate(rate):
    if isinstance(rate, bool):
        raise InvalidSampleRate("采样率必须是有限正数，不能是布尔值")
    if not isinstance(rate, (int, float, np.integer, np.floating)):
        try:
            rate = float(rate)
        except (TypeError, ValueError) as exc:
            raise InvalidSampleRate("采样率必须是有限正数") from exc
    rate = float(rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise InvalidSampleRate("采样率必须是有限正数")
    return rate


def _acf(a):
    n = a.size
    nfft = 1 << int(math.ceil(math.log2(max(2, 2 * n - 1))))
    spec = np.fft.rfft(a, nfft)
    return np.fft.irfft(spec * spec.conj(), nfft)[:n]


def _parab_min(y, k):
    """离散极小值 y[k] 的亚样本偏移量（|偏移| <= 0.5）。"""
    if k <= 0 or k + 1 >= len(y):
        return 0.0
    den = y[k - 1] - 2.0 * y[k] + y[k + 1]
    if den <= 0.0:
        return 0.0
    delta = 0.5 * (y[k - 1] - y[k + 1]) / den
    return delta if abs(delta) <= 0.5 else 0.0


# ---------------------------------------------------------------------------
# YIN
# ---------------------------------------------------------------------------

def _yin_f0(a, rate, lo, hi):
    """返回 (f0, 累积均值归一化差分最小值)。lag 范围 [lo, hi]。"""
    n = a.size
    r = _acf(a)
    energy = np.concatenate(([0.0], np.cumsum(a * a)))
    total = energy[-1]
    tau = np.arange(1, hi + 1)
    diff = energy[n - tau] + (total - energy[tau]) - 2.0 * r[1:hi + 1]
    diff = np.maximum(diff, 0.0)

    cmnd = np.ones(hi + 1)
    cumulative = np.cumsum(diff)
    idx = np.arange(1, hi + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmnd[1:] = np.where(cumulative > 1e-300, diff * idx / cumulative, 1.0)

    seg = cmnd[lo:hi + 1]
    threshold = min(0.30, float(seg.min()) * 1.15 + 0.02)
    pick = None
    for t in range(lo, hi + 1):
        left = cmnd[t - 1] if t > 1 else 1.0
        right = cmnd[t + 1] if t < hi else 1.0
        if cmnd[t] <= left and cmnd[t] <= right and cmnd[t] <= threshold:
            pick = t
            break
    if pick is None:
        pick = lo + int(np.argmin(seg))
    tau_ref = pick + _parab_min(diff, pick - 1)
    return rate / tau_ref, float(cmnd[pick])


# ---------------------------------------------------------------------------
# 频谱与谐波分析
# ---------------------------------------------------------------------------

def _spectrum(a, rate, pad=4):
    n = a.size
    nfft = 1 << int(math.ceil(math.log2(max(2, pad * n))))
    window = np.hanning(n) if n > 2 else np.ones(n)
    mag = np.abs(np.fft.rfft(a * window, nfft))
    return mag, rate / float(nfft)


def _peaks(mag, df, floor):
    """显著局部极大值 -> [(频率, 幅度)]，按幅度降序（向量化实现）。"""
    if mag.size < 3:
        return []
    logm = np.log(np.maximum(mag, 1e-300))
    left = mag[:-2]
    mid = mag[1:-1]
    right = mag[2:]
    is_peak = (mid >= left) & (mid > right) & (mid >= floor)
    idx = np.nonzero(is_peak)[0] + 1
    if idx.size == 0:
        return []
    den = logm[idx - 1] - 2.0 * logm[idx] + logm[idx + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(den < 0, 0.5 * (logm[idx - 1] - logm[idx + 1]) / den, 0.0)
    freqs = (idx + delta) * df
    amps = mag[idx]
    order = np.argsort(-amps)
    return [(float(freqs[j]), float(amps[j])) for j in order]


def _explain(peaks, f, nyq, df):
    """候选基频 f 对谱峰的解释度；返回 (score, 已解释峰, 谐波上限)。

    score = 被解释谱峰占比 - 0.5 * (缺失低次谐波占比)：

    * 纯音：唯一谱峰被解释、无缺失 -> 1.0；
    * 次谐波假设（把已有峰当成高次谐波）：会声明若干低次谐波位置却无峰，
      缺失惩罚随之上升，从而纠正倍频/半频歧义；
    * 基频缺失时，能解释最多谱峰的候选胜出（同分取更低频率）。
    """
    if f <= 0.0:
        return -1.0, [], 0
    kmax = min(12, int(nyq // f))
    if kmax < 1:
        return -1.0, [], 0
    explained = []
    for pf, pa in peaks:
        m = int(round(pf / f))
        if m < 1 or m > kmax:
            continue
        if abs(pf - m * f) <= max(0.012 * m * f, 2.5 * df):
            explained.append((m, pf, pa))
    if not explained:
        return 0.0, [], kmax
    fraction = len(explained) / float(len(peaks))
    mmax = max(e[0] for e in explained)
    matched = {e[0] for e in explained}
    missing = sum(1 for m in range(1, mmax + 1) if m not in matched)
    return fraction - 0.5 * missing / float(max(mmax, 1)), explained, kmax


def _dtft_peak(a, rate, f_c, rel=0.12):
    """Hann 加窗 DTFT 在 f_c 附近的主瓣极大值：FFT 粗定位 + 黄金分割精修。"""
    n = a.size
    if n <= 2:
        return f_c
    window = np.hanning(n)
    aw = a * window
    nfft = 1 << int(math.ceil(math.log2(max(8, 16 * n))))
    mag = np.abs(np.fft.rfft(aw, nfft))
    df = rate / float(nfft)
    k0 = int(round(f_c / df))
    span = max(4, int(math.ceil(rel * f_c / df)))
    klo = max(1, k0 - span)
    khi = min(len(mag) - 2, k0 + span)
    if khi <= klo:
        return f_c
    k = klo + int(np.argmax(mag[klo:khi + 1]))
    lo = (k - 1.5) * df
    hi = (k + 1.5) * df
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    t = np.arange(n, dtype=float)

    def magnitude(freq):
        basis = np.exp(-2j * math.pi * freq * t / rate)
        return float(np.abs(np.vdot(basis, aw)))

    x1 = hi - golden * (hi - lo)
    x2 = lo + golden * (hi - lo)
    f1, f2 = magnitude(x1), magnitude(x2)
    for _ in range(40):
        if hi - lo < 1e-9 * f_c:
            break
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + golden * (hi - lo)
            f2 = magnitude(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - golden * (hi - lo)
            f1 = magnitude(x1)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

class Pitch:
    """自相关基频估计器（无状态、可并发复用/多实例并发调用）。"""

    def estimate(self, samples, rate=None):
        """估计基频。

        参数
        ----
        samples : 实数一维序列。空序列或全零视为静音。
        rate    : 采样率（有限正数）。也接受 `estimate(samples)` 形式，
                  此时 `samples` 为 `(数据, 采样率)` 二元组或对象仅含数据。

        返回
        ----
        PitchResult : ``float`` 子类，数值为基频 Hz（静音/无基频为 0.0），
                      同时可解包为 ``(freq, confidence)``；
                      ``confidence`` ∈ [0, 1]，静音为 0.0。

        异常
        ----
        InvalidSampleRate : 采样率非数字、非有限或 <= 0。
        NonFiniteInput    : 样本包含 NaN/Inf（ValueError 与 TypeError 均可捕获）。
        InvalidSamples    : 样本不是一维实数序列。
        """
        samples, rate = self._unpack(samples, rate)
        rate = _check_rate(rate)
        a = _prepare(samples)
        if a is None:
            return PitchResult(0.0, 0.0)
        n = a.size
        lo = max(2, int(math.floor(rate / FMAX)))
        hi = min(int(math.ceil(rate / FMIN)), n // 2)
        if n < 16 or hi <= lo + 2:
            return PitchResult(0.0, 0.0)

        f_yin, _ = _yin_f0(a, rate, lo, hi)
        mag, df = _spectrum(a, rate)
        nyq = df * (len(mag) - 2)
        base = float(mag.max())
        if base <= 0.0:
            return PitchResult(0.0, 0.0)
        noise = float(np.median(mag))
        peaks = _peaks(mag, df, max(0.05 * base, 8.0 * noise))[:16]
        if not peaks:
            return PitchResult(0.0, 0.0)

        candidates = set()
        for ratio in (0.25, 0.5, 1.0, 2.0):
            f = f_yin * ratio
            if FMIN * 0.8 <= f <= FMAX * 1.2:
                candidates.add(round(f, 4))
        for pf, _ in peaks[:6]:
            for m in range(1, 7):
                f = pf / m
                if FMIN * 0.8 <= f <= FMAX * 1.2:
                    candidates.add(round(f, 4))

        best = None
        for f in sorted(candidates):
            score, explained, _kmax = _explain(peaks, f, nyq, df)
            if score <= 0.0 or not explained:
                continue
            key = (score, -f)  # 同分时取更低频率，避免倍频歧义
            if best is None or key > best[0]:
                best = (key, f, score, explained)
        if best is None or best[2] < 0.6:
            return PitchResult(0.0, 0.0)
        _key, f_hyp, score, explained = best

        fundamental = [e for e in explained if e[0] == 1]
        if fundamental:
            f_out = _dtft_peak(a, rate, fundamental[0][1])
            if abs(f_out - f_hyp) / f_hyp >= 0.03:
                f_out = f_hyp
        else:
            weights = np.array([(pa / base) / m for m, _, pa in explained])
            orders = np.array([m for m, _, _ in explained], dtype=float)
            observed = np.array([pf for _, pf, _ in explained])
            f_out = float((weights * orders * observed).sum()
                          / (weights * orders * orders).sum())

        tolerance = 0.005
        if not (FMIN * (1 - tolerance) <= f_out <= FMAX * (1 + tolerance)):
            return PitchResult(0.0, 0.0)
        f_out = min(max(f_out, FMIN), FMAX)

        confidence = self._confidence(a, rate, f_out, score)
        return PitchResult(f_out, confidence)

    @staticmethod
    def _confidence(a, rate, f_out, score):
        """置信度 = 谐波解释度 * 周期质量，均归一化到 [0, 1]。

        周期质量用基频 lag 处的归一化自相关，并按窗口内周期数归一化
        （纯音的 rho 约等于 1 - 1/C，C 为窗口周期数），因此同一纯音
        在短窗口和长窗口下的置信度一致。加噪后质量下降到约 0.75，
        但纯音的谐波解释度仍为 1，故强噪声下置信度仍 ≥ 视频面要求。
        """
        n = a.size
        lag = int(round(rate / f_out))
        r = _acf(a)
        rho = float(r[lag]) / max(float(r[0]), 1e-300) if 0 < lag < n else 0.0
        rho = min(1.0, max(0.0, rho))
        cycles = n / float(lag) if lag > 0 else 0.0
        expected = 1.0 - 1.0 / cycles if cycles > 1.0 else 0.0
        quality = rho / expected if expected > 0.05 else 0.0
        quality = min(1.0, max(0.0, quality))
        fit = min(1.0, max(0.0, (score - 0.6) / 0.3))
        return float(fit * quality)

    @staticmethod
    def _unpack(samples, rate):
        if rate is not None:
            return samples, rate
        if isinstance(samples, tuple) and len(samples) == 2:
            return samples[0], samples[1]
        raise InvalidSampleRate("缺少采样率参数")
