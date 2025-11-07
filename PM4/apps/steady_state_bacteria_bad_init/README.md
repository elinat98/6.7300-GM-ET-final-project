## Steady-state of bacterial model (poor initial guess)

Goal: Show that Newton may fail or stagnate when started far from a solution in a nonlinear system.

Run:

```bash
python PM4/apps/steady_state_bacteria_bad_init/solve.py
```

Expected behavior:
- Residual norm does not reliably decrease to tolerance within the iteration budget.
- Script prints non-convergence note. This motivates using better initialization, damping/homotopy, or pre-integration to a neighborhood of a steady state.



