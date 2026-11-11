# Proposal: funcpac-style linear Gaussian state-space models in dynestyx

Status: decisions 1 to 3 of section 7 accepted on 2026-11-11 (add `L`; keep
the positive definite guard as the default behind opt-in `allow_singular`;
native predict-first option plus a mapping constructor). Decisions 4 to 6
are open. Written 2026-11-11 on branch
`feature/lgssm-general-innovations` (cut from `main` at `99756490`, v0.8.1).
Nothing in `dynestyx/` has been changed yet; this directory holds only this
proposal and the evidence scripts under `evidence/`.

## 1. Goal

Let dynestyx evaluate the likelihood (and, later, smooth) the linear Gaussian
state-space model (LGSSM) as defined in funcpac's `StateSpace`
(`funcpac/lss/base.py`), including its structurally important special cases:

- fewer state shocks than states ($n_{\text{innov}} < n_{\text{states}}$), so the
  state noise covariance is singular;
- zero measurement noise ($\text{Cov}_{\text{obs}} = 0$);
- an initial state placed one step before the first observation.

The same model class is what pymfss (a port of MFSS) uses, with the symbols
$Z, H, T, R, Q$ in place of $C, \text{Cov}_{\text{obs}}, A, B, \text{Cov}_{\text{innov}}$.

Non-goals: diffuse initial states and mixed-frequency accumulators (pymfss),
particle or ensemble filters with $R = 0$ (a point-mass observation has no
density), and any change to dynestyx's default behaviour.

## 2. The two model forms

| | funcpac `StateSpace` | dynestyx `LTI_discrete` |
|---|---|---|
| State equation | $x_t = A_0 + A x_{t-1} + B u_t$, $u_t \sim N(0, \Sigma)$ | $x_{k+1} \sim N(A x_k + B_u u_k + b,\ Q)$ |
| Shape of noise | $B$ is $n \times g$, $\Sigma$ is $g \times g$, $g \le n$ allowed | $Q$ is $n \times n$, must be positive definite |
| Observation | $y_t = S + C x_t + v_t$, $v_t \sim N(0, \text{Cov}_{\text{obs}})$ | $y_k \sim N(H x_k + D u_k + d,\ R)$, $R$ positive definite |
| Default $\text{Cov}_{\text{obs}}$ / $R$ | zero | no default |
| Initial state | $x_0, P_0$ is the law one step BEFORE the first observation | $m_0, C_0$ is the law AT the first observation |
| Default initial covariance | stationary solution of the Lyapunov equation | identity |
| Missing data | mask | NaN entries, supported |

Timing was checked in the code of all three packages: funcpac's loop computes
`x_pred = A @ x + A0` before the first update
(`funcpac/likelihood/_kalman_numba.py`, line 390), pymfss has `a_init = T0 @ a0`
("Initial prediction", `filter_uni_jit.py`, line 85), and dynestyx's cuthbert
wrapper treats the first step as having no transition (`is_first_step` branch
returning `F = I, chol_Q = 0`).

The mapping between the conventions is
$m_0' = A x_0 + A_0$, $P_0' = A P_0 A^\top + B \Sigma B^\top$. Passing funcpac's
`x0` and `Cov_state0` to dynestyx unchanged gives a different likelihood
(-26.58376424 instead of -26.81025400 on the test model in `evidence/`).

## 3. Evidence collected

Everything below was run, with outputs saved in `evidence/output_*.txt`
(see `evidence/README.md` for how to rerun). Test model: $n = 2$, $p = 1$,
$A = \begin{pmatrix} 0.8 & 0.1 \\ 0 & 0.5 \end{pmatrix}$, $H = (1\ 0)$,
$B = (1\ 0)^\top$ (one shock), $\Sigma = 1$, $R = 0.1$, $T = 20$,
observations from `numpy.random.default_rng(0)`. The reference is a short
NumPy Kalman filter (`evidence/common.py`).

### 3.1 The numerical engines are not the obstacle

| Check | Result |
|---|---|
| cuthbertlib `predict`/`update` primitives, singular $Q$, $R = 0$, both conventions | match the NumPy reference to 1e-14 |
| Gradient of the log-likelihood through a `lax.scan` over those primitives, singular $Q$, with $R > 0$ and $R = 0$ | agrees with central differences to 2e-9 |
| cd_dynamax `lgssm_filter`, singular $Q$ (validation off) | matches the reference to 5e-8 or better; with $R = 0$ the difference is 1.4e-6 (cause not investigated) |
| cd_dynamax `lgssm_smoother`, singular $Q$ and singular $P_0$ (singular predicted covariance, validation off) | matches a pseudo-inverse RTS reference to 2.9e-9 |
| funcpac's own `kalman_filter` against the NumPy reference with the mapped initial condition | identical, -26.81025400 |

### 3.2 What fails today, and why

| Failure | Where | Cause (confirmed by running) |
|---|---|---|
| `ValueError: MultivariateNormal distribution got invalid covariance_matrix` for singular $Q$ or $R = 0$ | `DynamicalModel` construction | `LinearGaussianStateEvolution.__call__` / `LinearGaussianObservation.__call__` build a numpyro `MultivariateNormal`, and the construction-time probe in `models/checkers.py` evaluates it |
| cuthbert wrapper returns NaN for singular $Q$ (validation off) | `_kalman_dynamics_params_builder`, `discrete_filter.py` lines 760-778 | `jnp.linalg.cholesky(Q)` on a singular matrix |
| cuthbert returns NaN log-likelihood for $R = 0$, even sequentially and even with the predict-first convention | cuthbertlib `filtering_operator` (`kalman/filtering.py`) | `mu = cho_solve((U1, True), b1)` needs the previous element's factor `U1` non-singular. With $R = 0$ the filtered covariance is always singular (the observed component is known exactly), so `U1` is singular |
| cuthbert associative filter returns NaN log-likelihood for singular $Q$ | same line | element factors `U` are singular when $Q$ is. The filtered MEANS and Cholesky factors are finite and equal the sequential ones to 2e-16; only the accumulated log-likelihood is NaN, from the 4th step on |
| cuthbert smoother returns NaN when the predicted covariance is singular | `cuthbertlib/kalman/smoothing.py`, triangular solve against `Phi11` | needs $F P F^\top + Q$ non-singular. With a positive definite predicted covariance and singular $Q$ it matches the reference to 4e-16 |
| associative filter and smoother reject an $n \times g$ factor | `jnp.zeros_like(F)` and `jnp.eye(nx)` blocks | they need an $n \times n$ factor. The sequential filter accepts $n \times g$. Padding to $n \times n$ with zero columns works |
| Dummy first element degenerate when $Q = 0$ and $R = 0$ | `filter_prepare` on the `is_first_step` element | $S = 0$. This is a second, separate cause for $R = 0$ in the dynestyx convention |

### 3.3 Hypotheses that were wrong, kept so they are not retried

- "The predict-first convention removes the $R = 0$ failure." It does not: the
  `cho_solve` cause above remains. (Probe output, `predict` section.)
- "The associative filter works with a padded singular factor because its
  elements are finite." Elements are finite; the combination operator is not.

### 3.4 Other facts

- `DiagonalCovariance(sd=[1.0, 0.0])` and `ScalarCovariance(sd=0.0)` already
  construct (zeros are explicitly allowed by `_resolve_variance`, added in
  #410). The covariance types anticipate singular covariances; the failure is
  entirely downstream, in the `MultivariateNormal` construction.
- `relax_dynamics` (`dynestyx/relaxations.py`, #410) is the project's existing
  tool for turning degenerate components into non-degenerate Gaussians by
  adding noise. It is an approximate route to run funcpac-style models today. It
  was not tested for that purpose.
- The "singular or indefinite" guard in `discretization/numerics.py`
  (`_finalize_covariance`) came in with #299 (2026-08-17), has an explicit
  error message recommending `covariance_jitter`, and is not mentioned in the
  docs. It looks deliberate; whether it was discussed is unknown (no issue or
  PR text was read).
- The installed JAX (0.12.0.1) has no `jax.scipy.linalg.solve_discrete_lyapunov`.
  funcpac's solver (`lss/_lyapunov.py`) is NumPy/Numba.
- `relaxations._relax_state_evolution` adds noise to `evolution.cov` assuming it
  is $n \times n$. An `L` field would collide with it (section 5.1).

### 3.5 Not tested

- Missing observations on the scan path (the primitive `kf.update` contains
  `collect_nans_chol`, so it is expected to work).
- The simulators, posterior rollout and `observation_predictions` with a
  degenerate transition distribution (not read).
- Time-varying parameters on the scan path.
- Whether funcpac's concrete models (`LLSW`, `ML`, `bm`, `nyfed`,
  `imf_commodity`) produce a singular mapped $P_0'$ (singular $A$ together with
  singular $B\Sigma B^\top$ would).
- Performance of anything. No benchmark was run, and none is needed to
  decide the interface.
- Whether cuthbert's maintainers would take the two upstream changes in 3.2.

## 4. Proposal in three phases

Sizes are relative (S, M, L), not time estimates.

### Phase 1: filtering likelihood (size M)

This is enough for MLE and MCMC, which only need the likelihood.

1. Model layer. Add to `LinearGaussianStateEvolution` an optional loading
   matrix `L` ($n \times g$), appended as a new last field of
   `LinearGaussianParams` so that existing 4-tuple unpacking in `__call__`
   keeps working. When `L` is given, `cov` is the $g \times g$ shock
   covariance and the effective state noise is $L\,\text{cov}\,L^\top$. Add a
   static `allow_singular: bool = False` flag on `LinearGaussianStateEvolution`
   and `LinearGaussianObservation`. With the flag `False` nothing changes (the
   positive definite contract and the current error messages stay). With it
   `True`, `__call__` returns a small degenerate-Gaussian distribution (new
   module) that can sample `loc + L chol(cov) eps` and exposes `mean` and
   `covariance_matrix`, but whose `log_prob` raises a clear error. `models/
   checkers.py` must accept it in the construction probe.
2. Factor helper in `models/covariances.py`: return an $n \times n$ square-root
   factor, `pad(L @ chol(cov))` for the shock-space form and the elementwise
   square root for `Scalar`/`DiagonalCovariance`. Build factors this way, never
   by `cholesky` of an assembled singular matrix. Avoid an `eigh` based factor
   for `FullCovariance`: its gradient is unstable at repeated eigenvalues (not
   tested here, standard behaviour).
3. Filter path. For `KFConfig(filter_source="cuthbert")` on an
   `allow_singular` model, run a `lax.scan` over
   `cuthbertlib.kalman.filtering.predict` and `.update` instead of
   `cuthbert.filter`. Return per-step states in the same structure the
   wrappers already consume (`mean`, `chol_cov`, `log_normalizing_constant`),
   so `_cholesky_state_sequence_to_dists` and the recording options keep
   working. Raise a clear error for `associative=True` with a singular model.
   Evidence: section 3.1, 8 of 8 cases at machine precision.
4. cd_dynamax path. Let `_lti_to_lgssm_params` accept degenerate
   distributions (it already reads arrays, not distributions, for $Q$ and $R$).
   This path needs no new numerics, but is less precise ($R = 0$: 1.4e-6).
5. Initial condition, two options (decision 3 below):
   (a) a native option, name to be chosen, that makes the first step a regular
   transition (verified equal to funcpac); or
   (b) a constructor `LTI_discrete_from_innovations(A, B, Cov_innov, C,
   Cov_obs, x0, Cov_state0, ...)` that applies the mapping in section 2.
6. `stationary_initial_covariance(A, Sigma_state)` in JAX for the funcpac
   default of $P_0$. Candidates: a Kronecker solve of
   $(I - A \otimes A)\,\text{vec}(P) = \text{vec}(Q)$, or a doubling
   iteration. funcpac's `DiscreteLyapunovSolver` is the reference
   implementation to read. Requires a stable $A$; raise otherwise.
7. `relaxations.py`: when `L` is set, `relax_dynamics` must convert to the
   $n \times n$ form (`L cov L^T + addition`, `L = None`) before adding noise.
8. Guard behaviour for backends that need a density (particle filter,
   EnKF with $R = 0$, HMM): raise `NotImplementedError` naming the reason.

Acceptance: the eight cases of section 3.1 and gradient agreement, as
tests (section 6).

### Phase 2: smoothing (size S to M)

- Route `allow_singular` models to the cd_dynamax smoother, which handled the
  singular-predicted-covariance case (2.9e-9). Have the cuthbert smoother
  raise a clear error for them instead of returning NaN.
- Optional upstream: a pseudo-inverse or shock-space (disturbance) smoother
  in cuthbertlib, which is what pymfss implements. Not required.

### Phase 3: optional

- Parallel-in-time (associative) likelihood for singular $Q$. Needs the
  `cho_solve` in cuthbertlib's `filtering_operator` replaced. Only worth doing
  if long series on accelerators are a use case; the states are already
  correct, so only the log-likelihood term is missing.
- Singular $P_0$ (known initial states) beyond what the predict-first option
  gives automatically.

## 5. Interactions to handle

### 5.1 `L` and the Gaussian relaxation

`relax_dynamics(mode="add")` does `covariance_matrix(evolution.cov, state_dim)
+ addition`. With `L` present `cov` is $g \times g$, so this breaks. Resolve by
converting to the $n \times n$ form first (Phase 1, item 7).

### 5.2 Backwards compatibility

`allow_singular` defaults to `False`, `L` defaults to `None`, and
`LinearGaussianParams` gains a trailing field. Public behaviour for existing
models is unchanged. The cd_dynamax backend keeps its constant-parameter
restriction.

### 5.3 Callable (time-varying) parameters

`L` needs the same callable treatment as the other fields (`(t_now, t_next)`).
The scan path can call `params_at` per step. The cd_dynamax path cannot.

## 6. Test plan (as a delta on what exists)

Existing KF tests: `tests/test_filters.py` compares the associative filter with
the sequential one (`test_cuthbert_associative_kf_matches_sequential`), and the
science tests recover parameters statistically. In the lines of
`test_filters.py` I read, I saw no closed-form likelihood oracle; the other
KF test files were not searched.

Add `tests/test_lgssm_general.py`:

- NumPy reference filter (adapted from `evidence/common.py`) as the oracle;
  parametrize over {full-rank, singular $Q$} x {$R > 0$, $R = 0$} x {both
  first-step conventions}, plus missing observations (currently untested).
- Gradient against central differences for singular $Q$ with $R > 0$ and
  $R = 0$.
- Golden value from funcpac: -26.81025400 for the model in section 3 (hard-coded;
  dynestyx must not depend on funcpac). Regenerate with
  `evidence/crosscheck_funcpac.py`.
- Smoother: cd_dynamax route against the pseudo-inverse RTS reference with
  singular predicted covariance.
- Contract tests: with `allow_singular=False` a singular $Q$ still raises;
  particle and ensemble filters raise a clear error for a degenerate model.
- Enable `jax_enable_x64` for these tests. Every probe here ran with x64
  enabled; behaviour at float32 was not tested.

## 7. Decisions

Accepted on 2026-11-11: decisions 1, 2 and 3. Decisions 4 to 6 remain open.

1. ACCEPTED: Shock-space `L` (recommended) versus a positive semi-definite $n \times n$
   $Q$ only. `L` keeps the exact square-root factor, matches funcpac and
   pymfss one to one, and is cheaper to simulate; the cost is the
   `relaxations.py` coupling in 5.1. A PSD-only design needs a pivoted
   Cholesky or `eigh` for dense matrices.
2. ACCEPTED: Keep the positive definite guard as the default and add the opt-in
   `allow_singular` (recommended), or change the default. Premise to confirm:
   that the guard is deliberate. Evidence: an explicit error message and
   #299, but no discussion read.
3. ACCEPTED: Native predict-first option (recommended) plus the mapping constructor, or
   the mapping alone. The native option avoids forming $P_0'$ explicitly and
   lets the square-root predict step produce a singular factor naturally.
4. For the filter: scan over cuthbertlib primitives (recommended, tested),
   cd_dynamax with validation bypassed (least code, 1.4e-6 at $R = 0$), or a
   fix upstream in cuthbertlib (cleanest, outside this repository).
5. Whether Phase 3 (parallel-in-time likelihood) matters for the intended use.
6. Where the work lives: a branch on the fork (current), with a PR to
   `BasisResearch/dynestyx` only if wanted.

## 8. Risks

- cuthbert is a young dependency (0.1.1). Using its low-level primitives
  couples dynestyx to their signatures; pin and test.
- A scan path duplicates a little of what `cuthbert.filter` does and so must
  keep returning the same state structure.
- The degenerate distribution is a new public concept; its `log_prob` must
  fail loudly, never return a finite wrong number.
- Singular models exercise numerics at the boundary (for example
  $H P H^\top + R$ must stay positive definite for the likelihood to exist,
  which funcpac's non-strict mode handles by a pseudo-inverse and dynestyx
  would not). Specify the behaviour when it does not hold: raise or NaN.
- The mapped initial covariance for singular $A$ is unchecked (3.5).
