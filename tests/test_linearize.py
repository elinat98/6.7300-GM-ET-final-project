# tests/test_linearize.py
import numpy as np
import os, sys
from tools.SimpleSolver import SimpleSolver

try:
    from tools.linearize import linearize_f
except ModuleNotFoundError:
    # add project root and retry
    THIS_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from tools.linearize import linearize_f


def evalf_linear(x, p, u):
    """
    Linear test model:
        f(x,u) = A x + B u + c
    Accepts x as (N,) or (N,1) and returns same shape as input (1-D or column).
    p is expected to contain 'A','B','c'.
    """
    A = np.asarray(p['A'], dtype=float)
    B = np.asarray(p['B'], dtype=float)
    c = np.asarray(p['c'], dtype=float).ravel()
    x_arr = np.asarray(x, dtype=float)
    # accept column or flat:
    if x_arr.ndim == 2 and x_arr.shape[1] == 1:
        x_flat = x_arr.ravel()
        out = A.dot(x_flat) + B.dot(np.asarray(u).ravel()) + c
        return out.reshape((out.size,1))
    else:
        x_flat = x_arr.ravel()
        return A.dot(x_flat) + B.dot(np.asarray(u).ravel()) + c

def test_linearize_on_explicit_linear_model():
    # small test dimensions
    N = 4
    m_u = 2
    rng = np.random.RandomState(0)
    A_true = rng.randn(N, N) * 0.5
    B_true = rng.randn(N, m_u) * 0.3
    c_true = rng.randn(N) * 0.1

    p = {'A': A_true, 'B': B_true, 'c': c_true}
    x0 = np.linspace(1.0, 2.0, N)  # bias
    u0 = np.array([0.5, -0.2])

    # use linearize_f with no analytic jacobian -> FD will be used
    A_est, Ju_est, K0_est, B_est = linearize_f(evalf_linear, None, x0, p, u0, du=1e-6, fd_method='central')

    # checks
    assert A_est.shape == (N, N)
    assert Ju_est.shape == (N, m_u)
    # compare to true values
    # small tolerance because FD and tiny perturbations
    np.testing.assert_allclose(A_est, A_true, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(Ju_est, B_true, rtol=1e-5, atol=1e-6)
    # K0 = f(x0,u0) - A x0 - Ju u0 should equal c_true
    f0 = evalf_linear(x0, p, u0)
    # f0 might be 1-D, ensure shape
    f0 = np.asarray(f0).ravel()
    np.testing.assert_allclose(K0_est, c_true, rtol=1e-6, atol=1e-8)
