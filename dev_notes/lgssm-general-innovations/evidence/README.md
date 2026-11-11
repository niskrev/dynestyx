# Evidence for the funcpac-style LGSSM proposal

Collected 2026-11-11. The recorded versions are in `output_environment.txt`
(jax 0.12.0.1, numpyro 0.22.0, cuthbert and cuthbertlib 0.1.1, cd-dynamax
0.5.0, dynestyx 0.8.1 at `99756490`, funcpac `33e8076`, pymfss `12fb9cc`). Rerun after upgrading
any of these: the failures in the proposal are properties of those versions.

All scripts use one test model (`common.py`): $n = 2$, $p = 1$, a single shock
(rank-1 state noise), $T = 20$, data from `numpy.random.default_rng(0)`.
`common.py` also holds the NumPy reference Kalman filter and the
pseudo-inverse RTS smoother that every result is compared against.

| File | What it shows | Run from |
|---|---|---|
| `common.py` | Test model, NumPy reference filter and smoother | imported by the others |
| `probe_dynestyx_wrappers.py` | What today's dynestyx wrappers do with singular $Q$ and $R = 0$, with numpyro validation on and off; cd_dynamax smoother with a singular predicted covariance | dynestyx repo root |
| `probe_cuthbert_sqrt.py` | cuthbert's filter and smoother with rank-deficient factors: sequential and associative, both first-step conventions, rectangular factors, gradients, smoother; plus the `lax.scan` workaround over `kf.predict`/`kf.update` | dynestyx repo root |
| `crosscheck_funcpac.py` | funcpac's own likelihood against the reference under both initial-state conventions | funcpac environment |
| `output_*.txt` | Captured output of each script | n/a |

Commands:

```bash
cd /home/fourier/research/projects/basisresearch/dynestyx
uv run python dev_notes/lgssm-general-innovations/evidence/probe_dynestyx_wrappers.py
uv run python dev_notes/lgssm-general-innovations/evidence/probe_cuthbert_sqrt.py

cd /home/fourier/research/projects/funcpac
uv run --frozen python /home/fourier/research/projects/basisresearch/dynestyx/dev_notes/lgssm-general-innovations/evidence/crosscheck_funcpac.py
```

Type checking: the first three scripts pass `ty` in the dynestyx environment
when given as explicit file paths (the directory form finds no files). The
funcpac script needs funcpac on the path:
`uv run ty check --python <funcpac>/.venv crosscheck_funcpac.py`.

Notes for interpreting the outputs:

- A `nan` in a cuthbert row is a real failure of the log-normalizing
  constant, not of the filtered states; the script prints a separate
  "states ok" flag for the associative path.
- `probe_cuthbert_sqrt.py` uses cuthbert's raw state sequence, which has
  $T + 1$ entries (index 0 is the initial state). Compare against the
  reference with an offset of 1; an off-by-one here once produced a spurious
  1.9 error.
- The `noop` first-step convention is dynestyx's today; `predict` is funcpac's.
