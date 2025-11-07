## Steady-state of bacterial model (good initial guess)

Goal: Solve f(x,p,u)=0 for the bacterial model near a numerically obtained steady state. We first integrate with a small step using `SimpleSolver` to obtain a good initial guess, then apply Newton.

Run:

```bash
python PM4/apps/steady_state_bacteria_good_init/solve.py
```

Expected behavior:
- Residual norm decreases rapidly (quadratic convergence) to below tolerance.
- Final steady state equals the nearby fixed point of the ODE under constant inputs.



