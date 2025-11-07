## Steady-state with nonnegativity constraints (projected Newton)

We solve the steady-state equations `f(x,p,u)=0` of the bacterial model with bound constraints `x >= 0` (all components nonnegative).
At each Newton step we project the tentative update back to the feasible set by clamping negatives to zero, and we use backtracking line search on the projected trial point to ensure decrease in `||f||^2`.

Run:

```bash
python PM4/apps/steady_state_bacteria_projected/solve.py
```

Expected behavior:
- The script perturbs a near-steady initial guess to have some negative components.
- Projected Newton (with line search) enforces feasibility and typically converges to a valid steady state with all components `>= 0`.
- Output reports how many iterations required projections.

Notes:
- Projection can harm quadratic convergence near the solution if active constraints change; it is a robustness device to maintain feasibility, not a guarantee of convergence.
- For stricter constraints or better theory, consider barrier/penalty methods or interior-point variants.


