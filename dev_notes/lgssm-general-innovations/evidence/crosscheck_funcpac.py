"""Check the funcpac -> dynestyx initial-condition mapping against funcpac itself.

Run inside the funcpac environment (funcpac is not a dynestyx dependency):
    cd /home/fourier/research/projects/funcpac
    uv run --frozen python /abs/path/to/dev_notes/lgssm-general-innovations/evidence/crosscheck_funcpac.py

Prints funcpac's own Kalman log-likelihood next to the NumPy reference under
both initial-state conventions. The shifted convention should match funcpac.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Q_SINGULAR, R_VAR, SIGMA, A, H, L, kalman_filter, make_y  # noqa: E402
from funcpac.likelihood.kalman import kalman_filter as funcpac_filter  # noqa: E402
from funcpac.lss.base import StateSpace  # noqa: E402

y = make_y()
ss = StateSpace(
    A=A,
    B=L,
    C=H,
    Cov_innov=SIGMA,
    Cov_obs=R_VAR,
    x0=np.zeros((2, 1)),
    Cov_state0=np.eye(2),
)
fp = funcpac_filter(ss, y).log_likelihood
shifted = kalman_filter(
    A, Q_SINGULAR, H, R_VAR, np.zeros(2), np.eye(2), y, predict_first=True
)[0]
m0, P0 = A @ np.zeros(2), A @ np.eye(2) @ A.T + Q_SINGULAR
mapped = kalman_filter(A, Q_SINGULAR, H, R_VAR, m0, P0, y, predict_first=False)[0]
unshifted = kalman_filter(
    A, Q_SINGULAR, H, R_VAR, np.zeros(2), np.eye(2), y, predict_first=False
)[0]
print(f"funcpac kalman_filter log-likelihood                  {fp:.8f}")
print(f"NumPy ref, predict_first=True  (funcpac convention)   {shifted:.8f}")
print(f"NumPy ref, dynestyx convention with mapped (m0, P0)   {mapped:.8f}")
print(f"NumPy ref, dynestyx convention, x0/Cov_state0 as is   {unshifted:.8f}")
