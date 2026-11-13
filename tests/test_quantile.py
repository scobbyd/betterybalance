import numpy as np
from betterybalance.core import quantile as Q


def test_recovers_conditional_quantiles():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.uniform(-1, 1, n)
    y = 2 * x + rng.normal(0, 1, n)
    X = x[:, None]
    w50 = Q.fit_quantile(X, y, 0.5)
    w90 = Q.fit_quantile(X, y, 0.9)
    assert abs(w50[1] - 2) < 0.1 and abs(w50[0]) < 0.1
    assert abs(w90[0] - 1.2816) < 0.12  # z_0.9 of N(0,1)


def test_one_extreme_outlier_barely_moves_q50():
    rng = np.random.default_rng(3)
    n = 1000
    X = rng.uniform(-1, 1, (n, 1))
    y = 3 * X[:, 0] + rng.normal(0, 5, n)
    w = Q.fit_quantile(X, y, 0.5)
    y2 = y.copy()
    y2[0] = 5000.0
    w2 = Q.fit_quantile(X, y2, 0.5)
    assert abs(w2[0] - w[0]) < 1.0 and abs(w2[1] - w[1]) < 1.0


def test_standardise_uses_median_iqr_and_floors_constants():
    X = np.column_stack([np.arange(100.0), np.ones(100)])
    Xs, c, s = Q.standardise(X)
    assert c[0] == 49.5 and abs(s[0] - 49.5) < 1e-9 and s[1] == 1.0 and Xs[:, 1].max() == 0.0


def test_logistic_separates():
    rng = np.random.default_rng(4)
    n = 2000
    X = rng.normal(0, 1, (n, 2))
    y = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.3, n)) > 0
    w = Q.fit_logistic(X, y)
    p = Q.predict_proba(X, w)
    assert ((p > 0.5) == y).mean() > 0.9 and w[1] > 0 and w[2] < 0
