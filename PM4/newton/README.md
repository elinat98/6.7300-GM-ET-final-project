## NewtonSolver

Minimal Newton method for solving f(x,p,u)=0 with optional line search.

Usage example:

```python
from PM4.newton.NewtonSolver import newton_solve
from evalf_bacterial import evalf

p = {...}
x0 = ... # (N,1)
u = ...  # (m_u,)
x_star, info = newton_solve(evalf, x0, p, u, jacobian_fn=None, max_iter=50, tol=1e-8)
print(info)
```

If `jacobian_fn` is not provided, the solver tries the analytic Jacobian from `jacobian_tools.py` and otherwise falls back to finite differences from `tools/eval_Jf_FiniteDifference.py`.



