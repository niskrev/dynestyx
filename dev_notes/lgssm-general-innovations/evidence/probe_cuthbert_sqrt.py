"""cuthbert square-root Kalman filter/smoother with rank-deficient factors.

Run from the repo root:
    uv run python dev_notes/lgssm-general-innovations/evidence/probe_cuthbert_sqrt.py

Builds `kalman.build_filter` directly (as dynestyx's wrapper does) but supplies
square-root factors that need not be full rank, and checks each result against
the NumPy reference in `common.py`.

Two first-step conventions are compared:
  "noop"    dynestyx today: the initial law is the state at the first
            observation, so the first element has no transition (F=I, Q=0).
  "predict" funcpac/pymfss: the initial law is one step BEFORE the first
            observation, so every element, the first included, has a transition.
"""

import jax
import jax.numpy as jnp
import numpy as np
from common import (
    Q_SINGULAR,
    R_VAR,
    SIGMA,
    A,
    H,
    L,
    T,
    kalman_filter,
    make_y,
    rts_smoother_pinv,
)

jax.config.update("jax_enable_x64", True)

from cuthbert import filter as cuthbert_filter  # noqa: E402
from cuthbert import smoother as cuthbert_smoother  # noqa: E402
from cuthbert.gaussian import kalman  # noqa: E402

from dynestyx.inference.integrations.cuthbert.discrete_filter import CuthbertInputs  # noqa: E402

n, p = 2, 1
y = make_y()
A_J = jnp.asarray(A)


def pad_square(chol_Q):
    """Pad an (n, g) factor with zero columns to (n, n)."""
    chol_Q = jnp.asarray(chol_Q)
    return jnp.concatenate([chol_Q, jnp.zeros((n, n - chol_Q.shape[1]))], axis=1)


def build(A_, chol_Q, chol_R, m0, chol_P0, *, first):
    def dyn(mi):
        noop = (jnp.eye(n), jnp.zeros(n), jnp.zeros(chol_Q.shape))
        regular = (A_, jnp.zeros(n), chol_Q)
        if first == "predict":
            return regular
        return jax.lax.cond(mi.is_first_step, lambda mi: noop, lambda mi: regular, mi)

    def obs(mi):
        return jnp.asarray(H), jnp.zeros(p), chol_R, mi.y

    return kalman.build_filter(m0, chol_P0, dyn, obs)


def inputs():
    z = jnp.zeros
    return CuthbertInputs(
        y=jnp.asarray(y),
        u=z((T, 0)),
        u_prev=z((T, 0)),
        time=jnp.arange(T, dtype=float),
        time_prev=jnp.arange(T, dtype=float) - 1.0,
        is_first_step=jnp.arange(T) == 0,
    )


def run(A_, chol_Q, chol_R, m0, chol_P0, *, parallel, first):
    f = build(A_, chol_Q, chol_R, m0, chol_P0, first=first)
    # cuthbert returns T+1 states: index 0 is the initial state.
    return f, cuthbert_filter(f, inputs(), f.init_prepare(), parallel=parallel)


m0, chol_P0 = jnp.zeros(n), jnp.eye(n)
chol_R = jnp.sqrt(jnp.asarray(R_VAR))
CASES = {
    "full-rank Q, R > 0 (sanity)": (
        0.1 * np.eye(2),
        R_VAR,
        jnp.sqrt(0.1) * jnp.eye(n),
        chol_R,
    ),
    "singular Q, R > 0": (
        Q_SINGULAR,
        R_VAR,
        pad_square(jnp.asarray(L) @ jnp.linalg.cholesky(jnp.asarray(SIGMA))),
        chol_R,
    ),
    "full-rank Q, R = 0": (
        0.1 * np.eye(2),
        np.zeros((1, 1)),
        jnp.sqrt(0.1) * jnp.eye(n),
        jnp.zeros((1, 1)),
    ),
    "singular Q and R = 0": (
        Q_SINGULAR,
        np.zeros((1, 1)),
        pad_square(jnp.asarray(L) @ jnp.linalg.cholesky(jnp.asarray(SIGMA))),
        jnp.zeros((1, 1)),
    ),
}

for first, predict_first in (("noop", False), ("predict", True)):
    print(f"\n== filter log-likelihood, first-step convention = {first!r}")
    for name, (Q, R, cq, cr) in CASES.items():
        ref = kalman_filter(
            A, Q, H, R, np.zeros(2), np.eye(2), y, predict_first=predict_first
        )[0]
        out = []
        seq_mean = None
        for par in (False, True):
            _, s = run(A_J, cq, cr, m0, chol_P0, parallel=par, first=first)
            out.append(float(s.log_normalizing_constant[-1]))
            if not par:
                seq_mean = s.mean
            else:
                same = (
                    bool(jnp.all(jnp.isfinite(s.mean)))
                    and float(jnp.abs(s.mean - seq_mean).max()) < 1e-10
                )
                states_ok = "states ok" if same else "states BAD"
        print(
            f"{name:30s} ref={ref:.8f} seq={out[0]:.8f} assoc={out[1]:.8f}  "
            f"(assoc {states_ok})"
        )

print("\n== rectangular (n x g) factor, singular Q, R > 0, noop first step")
chol_Q_rect = jnp.asarray(L) @ jnp.linalg.cholesky(jnp.asarray(SIGMA))
ref = kalman_filter(
    A, Q_SINGULAR, H, R_VAR, np.zeros(2), np.eye(2), y, predict_first=False
)[0]
for tag, par in (("sequential", False), ("associative", True)):
    try:
        _, s = run(A_J, chol_Q_rect, chol_R, m0, chol_P0, parallel=par, first="noop")
        print(
            f"filter {tag:11s} loglik={float(s.log_normalizing_constant[-1]):.8f} (ref {ref:.8f})"
        )
    except Exception as e:
        print(f"filter {tag:11s} ERROR {type(e).__name__}: {str(e)[:90]}")

print(
    "\n== gradient of log-likelihood w.r.t. (a11, shock variance), padded factor, noop first step"
)


def loglik(theta, parallel):
    a11, s2 = theta
    A_ = A_J.at[0, 0].set(a11)
    cq = pad_square(jnp.asarray(L) * jnp.sqrt(s2))
    _, s = run(A_, cq, chol_R, m0, chol_P0, parallel=parallel, first="noop")
    return s.log_normalizing_constant[-1]


theta = jnp.array([0.8, 1.0])
for tag, par in (("sequential", False), ("associative", True)):
    g = jax.grad(lambda t, par=par: loglik(t, par))(theta)
    eps = 1e-6
    fd = jnp.array(
        [
            (loglik(theta + eps * e, par) - loglik(theta - eps * e, par)) / (2 * eps)
            for e in jnp.eye(2)
        ]
    )
    print(
        f"{tag:11s} autodiff={np.asarray(g)} central-diff={np.asarray(fd)} max abs diff={float(jnp.abs(g - fd).max()):.2e}"
    )

print(
    "\n== smoother (sequential, as in dynestyx's wrapper), singular Q; state 0 is cuthbert's initial state"
)
chol_Q_sing = CASES["singular Q, R > 0"][2]
for label, chol_P0_ in (
    ("P0 = I (predicted cov PD)", jnp.eye(n)),
    ("P0 singular (predicted cov singular)", jnp.diag(jnp.array([1.0, 0.0]))),
):
    _, states = run(
        A_J, chol_Q_sing, chol_R, m0, chol_P0_, parallel=False, first="noop"
    )
    P0 = np.asarray(chol_P0_ @ chol_P0_.T)
    _, ms, Ps = kalman_filter(
        A, Q_SINGULAR, H, R_VAR, np.zeros(2), P0, y, predict_first=False
    )
    ref = rts_smoother_pinv(A, Q_SINGULAR, ms, Ps)
    sf = kalman.build_smoother(lambda mi: (A_J, jnp.zeros(n), chol_Q_sing))
    sm = cuthbert_smoother(sf, states, model_inputs=None, parallel=False)
    err = max(np.abs(np.asarray(sm.mean[k + 1]) - ref[k]).max() for k in range(T))
    print(f"{label:40s} max |smoothed mean - pinv RTS ref| = {err:.3e}")

print(
    "\n== funcpac -> dynestyx initial-condition mapping (compare with crosscheck_funcpac.py)"
)
Cs0 = np.eye(2)
m0_s, P0_s = A @ np.zeros(2), A @ Cs0 @ A.T + Q_SINGULAR
cq = CASES["singular Q, R > 0"][2]
_, s = run(
    A_J,
    cq,
    chol_R,
    jnp.asarray(m0_s),
    jnp.linalg.cholesky(jnp.asarray(P0_s)),
    parallel=False,
    first="noop",
)
print(
    f"noop first step, MAPPED init (m0=A x0, P0=A C A' + Q)   loglik = {float(s.log_normalizing_constant[-1]):.8f}"
)
_, s = run(
    A_J,
    cq,
    chol_R,
    jnp.zeros(n),
    jnp.linalg.cholesky(jnp.asarray(Cs0)),
    parallel=False,
    first="predict",
)
print(
    f"predict first step, RAW init (x0, Cov_state0)          loglik = {float(s.log_normalizing_constant[-1]):.8f}"
)
_, s = run(
    A_J,
    cq,
    chol_R,
    jnp.zeros(n),
    jnp.linalg.cholesky(jnp.asarray(Cs0)),
    parallel=False,
    first="noop",
)
print(
    f"noop first step, RAW init (wrong convention)           loglik = {float(s.log_normalizing_constant[-1]):.8f}"
)


# -- Workaround candidate: lax.scan over cuthbertlib's predict/update primitives ---
# The cuthbert `filter` goes through `filtering_operator`, whose log-normalizing
# constant calls cho_solve against the previous element's factor U1 and so needs
# it non-singular. `kf.update` does not: it needs only chol(H P H' + R) to be
# invertible. A plain scan over the primitives avoids the operator entirely.
from cuthbertlib.kalman import filtering as kf  # noqa: E402


def scan_loglik(A_, chol_Q, chol_R, m0, chol_P0, *, predict_first):
    ys = jnp.asarray(y)

    def step(carry, inp):
        m, cP, ll = carry
        yk, do_predict = inp
        m, cP = jax.lax.cond(
            do_predict,
            lambda mc: kf.predict(mc[0], mc[1], A_, jnp.zeros(n), chol_Q),
            lambda mc: mc,
            (m, cP),
        )
        (m, cP), ell = kf.update(m, cP, jnp.asarray(H), jnp.zeros(p), chol_R, yk)
        return (m, cP, ll + ell), None

    flags = jnp.ones(T, dtype=bool) if predict_first else jnp.arange(T) > 0
    (_, _, ll), _ = jax.lax.scan(step, (m0, chol_P0, 0.0), (ys, flags))
    return ll


print("\n== workaround: lax.scan over kf.predict / kf.update (sequential)")
for predict_first in (False, True):
    for name, (Q, R, cq, cr) in CASES.items():
        ref = kalman_filter(
            A, Q, H, R, np.zeros(2), np.eye(2), y, predict_first=predict_first
        )[0]
        got = float(scan_loglik(A_J, cq, cr, m0, chol_P0, predict_first=predict_first))
        print(
            f"predict_first={predict_first!s:5s} {name:30s} ref={ref:.8f} scan={got:.8f} |diff|={abs(got - ref):.1e}"
        )

print(
    "\n== workaround: gradient through the scan (a11, shock variance), R = 0 and singular Q"
)


def scan_theta(theta, cr):
    a11, s2 = theta
    A_ = A_J.at[0, 0].set(a11)
    return scan_loglik(
        A_,
        pad_square(jnp.asarray(L) * jnp.sqrt(s2)),
        cr,
        m0,
        chol_P0,
        predict_first=False,
    )


for label, cr in (("R > 0", chol_R), ("R = 0", jnp.zeros((1, 1)))):
    g = jax.grad(lambda t, cr=cr: scan_theta(t, cr))(theta)
    fd = jnp.array(
        [
            (scan_theta(theta + 1e-6 * e, cr) - scan_theta(theta - 1e-6 * e, cr)) / 2e-6
            for e in jnp.eye(2)
        ]
    )
    print(
        f"singular Q, {label}: autodiff={np.asarray(g)} central-diff={np.asarray(fd)} max abs diff={float(jnp.abs(g - fd).max()):.2e}"
    )
