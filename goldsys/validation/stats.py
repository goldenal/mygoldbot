"""Statistical corrections for selection bias: PSR, DSR, minimum track record length, PBO.
References: Bailey & López de Prado (2012, 2014); Bailey, Borwein, López de Prado, Zhu (2014).
No scipy dependency: normal cdf/ppf implemented with math.erf and a rational approximation."""
from __future__ import annotations

import itertools
import math

import numpy as np

EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation, refined with one Newton step. Good to ~1e-9."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= 1 - plow:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _moments(r: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return 0.0, 0.0, 0.0, 3.0
    m = r.mean()
    s = r.std(ddof=1)
    if s == 0:
        return m, 0.0, 0.0, 3.0
    z = (r - m) / s
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    return float(m), float(s), skew, kurt


def sharpe_nonannual(r: np.ndarray) -> float:
    m, s, _, _ = _moments(r)
    return m / s if s > 0 else 0.0


def probabilistic_sharpe_ratio(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P[true SR > sr_benchmark] given the observed per-period returns (non-annualised SR)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    sr = sharpe_nonannual(r)
    _, _, skew, kurt = _moments(r)
    if n < 3:
        return 0.0
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """E[max SR] of n_trials independent strategies with zero true SR and SR variance sr_var."""
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(max(sr_var, 1e-12))
    return sd * ((1 - EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials) + EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe_ratio(r: np.ndarray, n_trials: int, sr_var_across_trials: float | None = None) -> dict:
    """DSR = PSR with the benchmark set to the expected maximum SR from n_trials.
    sr_var_across_trials: variance of (non-annualised) Sharpe across the trials run; if None we
    use the observed SR^2/2 as a conservative placeholder."""
    r = np.asarray(r, dtype=float)
    sr = sharpe_nonannual(r)
    if sr_var_across_trials is None:
        sr_var_across_trials = max(sr ** 2 / 2, 1e-6)
    sr0 = expected_max_sharpe(max(int(n_trials), 1), sr_var_across_trials)
    psr = probabilistic_sharpe_ratio(r, sr0)
    return {"sharpe_per_period": sr, "n_trials": int(n_trials), "expected_max_sharpe": sr0, "dsr": psr}


def min_track_record_length(r: np.ndarray, sr_benchmark: float = 0.0, confidence: float = 0.95) -> float:
    """Minimum number of periods for PSR to reach `confidence`. inf if SR <= benchmark."""
    sr = sharpe_nonannual(r)
    _, _, skew, kurt = _moments(r)
    if sr <= sr_benchmark:
        return math.inf
    z = norm_ppf(confidence)
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (z / (sr - sr_benchmark)) ** 2


def probability_of_backtest_overfitting(M: np.ndarray, n_groups: int = 16, metric=None) -> dict:
    """PBO via combinatorially symmetric cross-validation (CSCV).
    M: (T periods x N trials) matrix of per-period returns for every trial.
    Splits T into n_groups blocks, for each half-combination finds the best IS trial and records
    its OOS rank; PBO = fraction of combinations where the IS-best is below median OOS."""
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if N < 2 or T < n_groups * 2:
        return {"pbo": float("nan"), "n_combinations": 0, "logits": []}
    metric = metric or (lambda X: np.nan_to_num(X.mean(axis=0) / (X.std(axis=0, ddof=1) + 1e-12)))
    groups = np.array_split(np.arange(T), n_groups)
    logits = []
    half = n_groups // 2
    for combo in itertools.combinations(range(n_groups), half):
        is_idx = np.concatenate([groups[g] for g in combo])
        oos_idx = np.concatenate([groups[g] for g in range(n_groups) if g not in combo])
        is_perf = metric(M[is_idx])
        oos_perf = metric(M[oos_idx])
        best = int(np.argmax(is_perf))
        rank = (oos_perf < oos_perf[best]).sum() / (N - 1) if N > 1 else 0.5  # relative rank in [0,1]
        w = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": int(len(logits)), "mean_logit": float(logits.mean())}
