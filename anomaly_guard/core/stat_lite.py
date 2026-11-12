"""Chi-square and Student-t helpers without SciPy.

The published app cannot ship SciPy. These match scipy.stats.chisquare and
scipy.stats.t.ppf closely enough for scoring and interval fits.
"""

from __future__ import annotations

import math

import numpy as np


def _regularized_gamma_q(a: float, x: float) -> float:
    """Upper regularized gamma Q(a, x) = Gamma(a, x) / Gamma(a)."""
    if x <= 0:
        return 1.0
    if a <= 0:
        return 0.0
    if x < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gamma_p_series(a, x)))
    return max(0.0, min(1.0, _gamma_q_continued(a, x)))


def _gamma_p_series(a: float, x: float) -> float:
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(400):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_continued(a: float, x: float) -> float:
    """Lentz continued fraction for the upper incomplete gamma."""
    tiny = 1e-30
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 400):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chisquare(
    observed: np.ndarray | None = None,
    expected: np.ndarray | None = None,
    *,
    f_obs: np.ndarray | None = None,
    f_exp: np.ndarray | None = None,
) -> tuple[float, float]:
    """Pearson chi-square statistic and survival p-value (df = k - 1)."""
    obs = np.asarray(f_obs if f_obs is not None else observed, dtype=float)
    exp = np.asarray(f_exp if f_exp is not None else expected, dtype=float)
    stat = float(np.sum((obs - exp) ** 2 / np.maximum(exp, 1e-15)))
    df = max(int(obs.size) - 1, 1)
    p_value = _regularized_gamma_q(df / 2.0, stat / 2.0)
    return stat, float(p_value)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    am = 1.0
    a0 = 1.0
    b0 = 1.0
    a1 = 1.0
    b1 = 1.0 - (a + b) * x / (a + 1.0)
    for m in range(1, 200):
        az = 2 * m
        d = m * (b - m) * x / ((a + az - 1.0) * (a + az))
        ap = a1 + d * a0
        bp = b1 + d * b0
        d = -(a + m) * (a + b + m) * x / ((a + az) * (a + az + 1.0))
        app = ap + d * a1
        bpp = bp + d * b1
        a0, b0, a1, b1 = ap, bp, app, bpp
        if abs(bpp) < 1e-30:
            bpp = 1e-30
        am = a1 / bpp
        a0 /= bpp
        b0 /= bpp
        a1 /= bpp
        b1 = 1.0
        if abs(am - a0 / max(abs(a0), 1e-30) * am) < 1e-12 and m > 2:
            # convergence checked via successive multipliers below
            pass
        if m > 2 and abs(1.0 - (a0 / a1 if a1 else 0.0)) < 1e-10:
            break
    return am


def _regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    return 1.0 - (math.exp(b * math.log(1.0 - x) + a * math.log(x) - ln_beta) / b) * _betacf(b, a, 1.0 - x)


def _student_t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    ib = _regularized_beta(a, b, x)
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t. p is the lower-tail probability."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    # Start from a normal approximation, then Newton on the CDF.
    z = _normal_ppf(p)
    guess = z + (z * z * z + z) / (4.0 * df)
    for _ in range(40):
        cdf = _student_t_cdf(guess, df)
        # density of t
        log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
        pdf = math.exp(log_c - (df + 1) / 2 * math.log(1 + guess * guess / df))
        if pdf < 1e-14:
            break
        step = (cdf - p) / pdf
        guess -= step
        if abs(step) < 1e-10:
            break
    return float(guess)


def _normal_ppf(p: float) -> float:
    """Acklam approximation of the standard normal quantile."""
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )
