"""dft 核心 API（纯标准库实现）。

- :meth:`Dft.transform` 采用混合基 Cooley-Tukey FFT，支持任意正长度
  （包括非 2 的幂）；小素数长度直接按朴素 DFT 定义计算，大素数长度走
  Bluestein chirp-z 化为 2 的幂卷积，因此结果始终与朴素定义

      X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)

  完全一致（float64 数值误差量级），且任意长度都不会退化为 O(N^2)。
- :meth:`Dft.window` 提供对称 Hann / Hamming 窗，端点取值精确。

全部计算只使用函数局部状态，实例无可变共享状态，天然线程安全。
"""

import cmath
import math

__all__ = ["Dft"]

_TWO_PI = 2.0 * math.pi


def _smallest_prime_factor(n):
    """返回 n（n >= 2）的最小素因子；n 本身为素数时返回 n。"""
    if n % 2 == 0:
        return 2
    p = 3
    while p * p <= n:
        if n % p == 0:
            return p
        p += 2
    return n


def _naive_dft(a):
    """朴素 DFT 定义，O(N^2)，作为素数长度回退与语义基准。"""
    n = len(a)
    out = [0j] * n
    for k in range(n):
        s = 0j
        # W_n^k 随 n 递推，避免每个采样点都调用一次 exp
        wk = cmath.exp(-1j * _TWO_PI * k / n)
        w = 1 + 0j
        for x in a:
            s += w * x
            w *= wk
        out[k] = s
    return out


def _ifft(a):
    """逆 FFT，借助共轭复用正向 _fft：ifft(x) = conj(fft(conj(x)))/N。"""
    n = len(a)
    spec = _fft([x.conjugate() for x in a])
    return [x.conjugate() / n for x in spec]


# 素数长度不超过该值时直接用朴素定义；更大的素数走 Bluestein，避免 O(N^2)
_BLUESTEIN_THRESHOLD = 256


def _bluestein(a):
    """Bluestein / chirp-z 算法：把任意长度 N 的 DFT 化为长度 M（2 的幂，
    M >= 2N-1）的循环卷积，整体复杂度 O(M log M)，用于大素数长度。"""
    n = len(a)
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    am = [0j] * m
    bm = [0j] * m
    # chirp 序列 c[j] = exp(pi*i*j^2/N)，且 c[-j] = c[j]
    for j in range(n):
        chirp = cmath.exp(1j * math.pi * j * j / n)
        am[j] = a[j] / chirp
        bm[j] = chirp
    for j in range(1, n):
        bm[m - j] = bm[j]
    fa = _fft(am)
    fb = _fft(bm)
    conv = _ifft([u * v for u, v in zip(fa, fb)])
    return [conv[k] * cmath.exp(-1j * math.pi * k * k / n) for k in range(n)]


def _fft(a):
    """混合基 DIT Cooley-Tukey FFT，输入/输出均为长度相同的 complex 列表。"""
    n = len(a)
    if n == 1:
        return [complex(a[0])]
    r = _smallest_prime_factor(n)
    if r == n:
        # 素数长度无法分治：小长度按朴素定义，大长度走 Bluestein
        if n <= _BLUESTEIN_THRESHOLD:
            return _naive_dft(a)
        return _bluestein(a)
    m = n // r
    # 按模 r 拆成 r 路长度为 m 的子序列并递归变换
    parts = [_fft(a[j::r]) for j in range(r)]
    out = [0j] * n
    for k in range(n):
        km = k % m
        base = cmath.exp(-1j * _TWO_PI * k / n)  # W_n^k
        w = 1 + 0j
        s = 0j
        for j in range(r):
            s += w * parts[j][km]  # W_n^(j*k) * A_j[k mod m]
            w *= base
        out[k] = s
    return out


class Dft:
    def transform(self, samples):
        """计算正向 DFT（非归一化），返回与输入等长的 complex 列表。

        参数:
            samples: 数值序列（实数或复数）。

        异常:
            TypeError: 输入不是数值序列、或元素不是数值。
            ValueError: 序列为空、或包含 NaN / Infinity 等非有限值。
        """
        if isinstance(samples, (str, bytes)) or not hasattr(samples, "__iter__"):
            raise TypeError("samples must be an iterable sequence of numeric values")

        data = []
        for v in samples:
            try:
                z = complex(v)
            except (TypeError, ValueError):
                # complex("x") 抛 ValueError、complex(object()) 抛 TypeError，
                # 这里统一归类为“元素类型非法”
                raise TypeError("all samples must be real or complex numbers")
            if math.isnan(z.real) or math.isnan(z.imag):
                raise ValueError("samples must not contain NaN")
            if math.isinf(z.real) or math.isinf(z.imag):
                raise ValueError("samples must not contain infinity")
            data.append(z)

        if not data:
            raise ValueError("samples must not be empty")

        return _fft(data)

    def window(self, n, kind="hann"):
        """生成长度为 n 的对称窗函数，返回 float 列表。

        参数:
            n: 窗长度，必须为正整数。
            kind: ``'hann'``（默认）或 ``'hamming'``。

        异常:
            TypeError: n 不是整数、或 kind 不是字符串。
            ValueError: n 非正、或 kind 不是受支持的窗类型。
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("window length n must be an integer")
        if n <= 0:
            raise ValueError("window length n must be positive")
        if not isinstance(kind, str):
            raise TypeError("window kind must be a string")
        if kind == "hann":
            if n == 1:
                return [1.0]
            w = [0.5 * (1.0 - math.cos(_TWO_PI * k / (n - 1))) for k in range(n)]
            # 端点按定义恰好为 0，避免余弦求值的舍入残差
            w[0] = 0.0
            w[-1] = 0.0
            return w
        if kind == "hamming":
            if n == 1:
                return [1.0]
            return [0.54 - 0.46 * math.cos(_TWO_PI * k / (n - 1)) for k in range(n)]
        raise ValueError(f"unknown window kind: {kind!r} (supported: 'hann', 'hamming')")
