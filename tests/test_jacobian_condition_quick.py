import numpy as np
import pytest
from jacobian_tools import evaljacobianf
from evalf_bacterial import evalf
from tools.SimpleSolver import SimpleSolver

def mk_params(m=3):
    return {
        'Q': np.eye(m),
        'rmax': np.array([1.0, 0.8, 0.6]),
        'K': np.array([0.5, 0.4, 0.3]),
        'alpha': np.array([0.1, 0.1, 0.1]),
        'd0': np.array([0.2, 0.15, 0.1]),
        'IC50': np.array([1.0, 0.8, 1.2]),
        'h': np.array([1.0, 1.0, 1.0]),
        'kC': 0.05
    }

def test_jacobian_condition_at_final_state_is_finite_and_reasonable():
    """Quick smoke test: Jacobian at final state has finite singular values and moderate cond number."""
    m = 3
    p = mk_params(m)
    x0 = np.array([10.0, 5.0, 2.0, 1.0, 0.2]).reshape((-1,1))
    def u(t): return np.array([0.5, 0.1])

    # run a short solver (small NumIter so fast)
    X, t = SimpleSolver(evalf, x0, p, u, NumIter=80, w=0.01)
    x_final = X[:, -1].ravel()

    J = evaljacobianf(x_final, p, u(t[-1]))
    # compute singular values
    s = np.linalg.svd(J, compute_uv=False)
    assert np.all(np.isfinite(s)), "Non-finite singular values in Jacobian"
    cond_num = s[0] / (s[-1] + 1e-300)
    # set a generous threshold so CI isn't overly brittle; tune as you like
    assert cond_num < 1e12, f"Condition number too large: {cond_num:.3e} (may indicate near-singularity)"
