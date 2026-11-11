"""Shared test model and NumPy reference filter/smoother for the LGSSM probes.

Convention used by dynestyx: the initial distribution is the law of the state
at the FIRST observation time, so the first observation is an update with no
preceding predict step (`predict_first=False`). funcpac and pymfss predict one
step before the first update (`predict_first=True`).
"""

import numpy as np

A = np.array([[0.8, 0.1], [0.0, 0.5]])
H = np.array([[1.0, 0.0]])
L = np.array([[1.0], [0.0]])  # loading: one shock, two states
SIGMA = np.array([[1.0]])  # shock covariance (g x g)
R_VAR = np.array([[0.1]])
Q_SINGULAR = L @ SIGMA @ L.T  # rank 1 in a 2-state system
T = 20


def make_y() -> np.ndarray:
    """Deterministic observations, shape (T, 1), identical in every probe."""
    return np.random.default_rng(0).standard_normal((T, 1))


def kalman_filter(A, Q, H, R, m0, P0, y, *, predict_first):
    """Textbook Kalman filter. Returns (loglik, filtered means, filtered covs)."""
    m, P = np.asarray(m0, float), np.asarray(P0, float)
    ll, ms, Ps = 0.0, [], []
    for k in range(len(y)):
        if predict_first or k > 0:
            m, P = A @ m, A @ P @ A.T + Q
        S = H @ P @ H.T + R
        v = y[k] - H @ m
        ll += -0.5 * (
            np.linalg.slogdet(2 * np.pi * S)[1] + (v @ np.linalg.solve(S, v)).item()
        )
        K = P @ H.T @ np.linalg.inv(S)
        m, P = m + K @ v, P - K @ S @ K.T
        ms.append(m)
        Ps.append(P)
    return ll, ms, Ps


def rts_smoother_pinv(A, Q, ms, Ps):
    """RTS smoother means using a pseudo-inverse of the predicted covariance."""
    out = [None] * len(ms)
    out[-1] = ms[-1]
    for k in range(len(ms) - 2, -1, -1):
        Pp = A @ Ps[k] @ A.T + Q
        G = Ps[k] @ A.T @ np.linalg.pinv(Pp)
        out[k] = ms[k] + G @ (out[k + 1] - A @ ms[k])
    return out
