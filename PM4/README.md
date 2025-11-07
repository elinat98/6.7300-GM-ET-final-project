## PM4: Newton solver and application demos

This module provides a basic Newton solver for solving nonlinear systems f(x,p,u)=0 and a set of small applications demonstrating convergence behavior under good and poor initial guesses.

### Layout

- `newton/` — core Newton method implementation
- `apps/` — example problems with per-app READMEs
  - `steady_state_bacteria_good_init/` — Newton converges quickly when initialized near a steady state
  - `steady_state_bacteria_bad_init/` — Newton struggles/fails with a poor initial guess
  - `scalar_toy_multiple_roots/` — scalar example with multiple roots and sensitivity to initialization

### Quick start

From repo root:

```bash
python PM4/apps/steady_state_bacteria_good_init/solve.py
python PM4/apps/steady_state_bacteria_bad_init/solve.py
python PM4/apps/scalar_toy_multiple_roots/solve.py
```

Each app prints convergence history and whether Newton converged (and to what).

### Notes

- The solver uses the analytic Jacobian when provided; otherwise it falls back to finite differences (via `tools/eval_Jf_FiniteDifference.py`).
- Optional backtracking line search improves robustness but does not guarantee convergence.



