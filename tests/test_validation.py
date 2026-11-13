import numpy as np
from goldsys.validation.stats import deflated_sharpe_ratio, probability_of_backtest_overfitting, norm_ppf


def test_norm_ppf():
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-5


def test_dsr_decreases_with_trials():
    r = np.random.default_rng(0).normal(0.0005, 0.01, 500)
    assert deflated_sharpe_ratio(r, 1)["dsr"] > deflated_sharpe_ratio(r, 200)["dsr"]


def test_pbo_noise_vs_edge():
    rng = np.random.default_rng(0)
    M = rng.normal(0, 0.01, (400, 20))
    assert 0.3 < probability_of_backtest_overfitting(M, 8)["pbo"] < 0.7
    M[:, 3] += 0.004
    assert probability_of_backtest_overfitting(M, 8)["pbo"] < 0.1
