"""What the current dynestyx Kalman wrappers do with rank-deficient Q and R = 0.

Run from the repo root:
    uv run python dev_notes/lgssm-general-innovations/evidence/probe_dynestyx_wrappers.py

Pass 1 uses numpyro's default argument validation; pass 2 disables it. Every
filter value is compared with the NumPy reference in `common.py`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from common import (
    Q_SINGULAR,
    R_VAR,
    A,
    H,
    T,
    kalman_filter,
    make_y,
    rts_smoother_pinv,
)

jax.config.update("jax_enable_x64", True)

from cd_dynamax.dynamax.linear_gaussian_ssm.inference import lgssm_smoother  # noqa: E402

from dynestyx import LTI_discrete  # noqa: E402
from dynestyx.inference.configs.filter import KFConfig  # noqa: E402
from dynestyx.inference.integrations.cd_dynamax.discrete_filter import (  # noqa: E402
    _lti_to_lgssm_params,
    compute_cd_dynamax_discrete_filter,
)
from dynestyx.inference.integrations.cuthbert.discrete_filter import (  # noqa: E402
    compute_cuthbert_filter,
)

y = make_y()
times = jnp.arange(T, dtype=float)
CASES = {
    "full-rank Q, R > 0": (0.1 * np.eye(2), R_VAR),
    "singular Q (1 shock, 2 states), R > 0": (Q_SINGULAR, R_VAR),
    "full-rank Q, R = 0": (0.1 * np.eye(2), np.zeros((1, 1))),
}


def one_pass(label: str) -> None:
    print(f"\n== {label}")
    for name, (Q, R) in CASES.items():
        ref = kalman_filter(A, Q, H, R, np.zeros(2), np.eye(2), y, predict_first=False)[
            0
        ]
        try:
            model = LTI_discrete(
                A=jnp.array(A), Q=jnp.array(Q), H=jnp.array(H), R=jnp.array(R)
            )
        except Exception as e:
            print(f"{name:40s} CONSTRUCTION ERROR {type(e).__name__}: {str(e)[:70]}")
            continue
        cd = compute_cd_dynamax_discrete_filter(
            model,
            KFConfig(filter_source="cd_dynamax"),
            obs_times=times,
            obs_values=jnp.array(y),
        ).marginal_loglik
        cu, _ = compute_cuthbert_filter(
            model,
            KFConfig(filter_source="cuthbert"),
            obs_times=times,
            obs_values=jnp.array(y),
        )
        print(
            f"{name:40s} ref={ref:.8f} cd_dynamax={float(cd):.8f} cuthbert={float(cu):.8f}"
        )


one_pass("pass 1: numpyro validation ON (default)")
numpyro.enable_validation(False)
one_pass("pass 2: numpyro validation OFF")

# cd_dynamax smoother when the predicted covariance is singular (known x2_0).
print("\n== cd_dynamax smoother, singular Q and singular P0 (validation off)")
P0 = jnp.diag(jnp.array([1.0, 0.0]))
model = LTI_discrete(
    A=jnp.array(A),
    Q=jnp.array(Q_SINGULAR),
    H=jnp.array(H),
    R=jnp.array(R_VAR),
    initial_cov=P0,
)
post = lgssm_smoother(
    _lti_to_lgssm_params(model), jnp.array(y), inputs=jnp.zeros((T, 0))
)
_, ms, Ps = kalman_filter(
    A, Q_SINGULAR, H, R_VAR, np.zeros(2), np.diag([1.0, 0.0]), y, predict_first=False
)
ref = rts_smoother_pinv(A, Q_SINGULAR, ms, Ps)
err = max(np.abs(np.asarray(post.smoothed_means[k]) - ref[k]).max() for k in range(T))
print(f"max |cd_dynamax smoothed mean - pinv RTS reference| = {err:.3e}")
