"""Quantile regression and logistic regression in plain numpy.

Quantile fit: majorise-minimise on the smoothed check loss (Hunter and Lange, 2000).
With residual r = y - Xw and e_i = max(|r_i|, eps), the check loss
rho_q(r) = r (q - 1[r < 0]) is majorised at r_i by
r^2 / (4 e_i) + (2q - 1) r / 2 + e_i / 4.
Minimising the sum over w gives X'W(y - Xw) + (2q - 1) X'1 = 0 with W = diag(1 / e_i),
so each step solves (X'WX + l2 D) w = X'W (y + (2q - 1) e). D leaves the intercept free.
"""

from __future__ import annotations

import numpy as np

_SCALE_FLOOR = 1e-6
_LOGIT_CLIP = 30.0


def _iqr(a: np.ndarray, axis: int = 0) -> np.ndarray:
    q75, q25 = np.percentile(a, [75, 25], axis=axis)
    return q75 - q25


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Xs, center, scale) using column median and IQR."""
    X = np.asarray(X, dtype=np.float64)
    center = np.median(X, axis=0)
    scale = _iqr(X, axis=0)
    scale = np.where(scale < _SCALE_FLOOR, 1.0, scale)
    return (X - center) / scale, center, scale


def apply_standardise(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=np.float64) - center) / scale


def _design(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([np.ones(X.shape[0]), X])


def _ridge_diag(p1: int, l2: float) -> np.ndarray:
    d = np.full(p1, l2)
    d[0] = 0.0
    return d


def fit_quantile(
    X: np.ndarray,
    y: np.ndarray,
    q: float,
    l2: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """Linear quantile regression. Returns weights, intercept first."""
    A = _design(X)
    y = np.asarray(y, dtype=np.float64)
    eps = max(1e-3 * float(_iqr(y)), 1e-6)
    ridge = np.diag(_ridge_diag(A.shape[1], l2))
    w = np.linalg.solve(A.T @ A + ridge, A.T @ y)
    for _ in range(max_iter):
        e = np.maximum(np.abs(y - A @ w), eps)
        wt = 1.0 / e
        target = y + (2.0 * q - 1.0) * e
        w_new = np.linalg.solve((A * wt[:, None]).T @ A + ridge, A.T @ (wt * target))
        done = float(np.max(np.abs(w_new - w))) < tol
        w = w_new
        if done:
            break
    return w


def predict_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _design(X) @ w


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -_LOGIT_CLIP, _LOGIT_CLIP)))


def fit_logistic(
    X: np.ndarray, y_bool: np.ndarray, l2: float = 1e-3, max_iter: int = 50
) -> np.ndarray:
    """Ridge logistic regression by Newton steps. Returns weights, intercept first."""
    A = _design(X)
    y = np.asarray(y_bool, dtype=np.float64)
    ridge_d = _ridge_diag(A.shape[1], l2)
    w = np.zeros(A.shape[1])
    for _ in range(max_iter):
        p = _sigmoid(A @ w)
        grad = A.T @ (p - y) + ridge_d * w
        s = np.maximum(p * (1.0 - p), 1e-9)
        H = (A * s[:, None]).T @ A + np.diag(ridge_d) + 1e-12 * np.eye(A.shape[1])
        step = np.linalg.solve(H, grad)
        w = w - step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return w


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _sigmoid(_design(X) @ w)


def pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    """Mean pinball loss."""
    d = np.asarray(y, dtype=np.float64) - np.asarray(yhat, dtype=np.float64)
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))
