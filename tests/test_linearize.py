# tests/test_linearize.py
import numpy as np
import numpy.testing as npt

from linearize import linearize_f
# external FD function used internally by linearize_f lives in tools; ensure package import path is correct in your project.

def eval_linear_model(x, p, u):
    """
    A simple explicit linear model that accepts either 1-D x or column (N,1).
    f(x,u) = A_true @ x + B_true[:,1:] @ u + B_true[:,0]
    """
    A = p['A']
    B = p['B']  # shape (N, 1+m_u) arranged [K0 | Ju]
    # normalize x to 1-D
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.ravel()
    u = np.asarray(u, dtype=float).ravel()
    # compute
    K0 = B[:, 0]
    Ju = B[:, 1:]
    return A.dot(x) + Ju.dot(u) + K0


def test_linearize_on_explicit_linear_model():
    rng = np.random.RandomState(1)
    N = 5
    m_u = 2

    # pick some exact matrices
    A_true = rng.randn(N, N) * 0.5
    Ju_true = rng.randn(N, m_u) * 0.3
    K0_true = rng.randn(N) * 0.1
    B_true = np.hstack([K0_true.reshape(N, 1), Ju_true])

    p = {'A': A_true, 'B': B_true}

    # choose bias x0 and u0
    x0 = np.linspace(1.0, 2.0, N)          # 1-D vector
    u0 = np.array([0.5, -0.2])

    # call linearize_f with no analytic J (so it must use external FD)
    A_est, B_est = linearize_f(eval_linear_model, J_eval=None, x0=x0, p=p, u0=u0,
                               du=1e-6, fd_method='central')

    # checks: shapes
    assert A_est.shape == (N, N)
    assert B_est.shape == (N, 1 + m_u)

    # A must match A_true closely (exact linear case -> exact up to FD rounding)
    npt.assert_allclose(A_est, A_true, rtol=1e-5, atol=1e-6)

    # K0 and Ju must match
    npt.assert_allclose(B_est[:, 0], K0_true, rtol=1e-6, atol=1e-8)
    npt.assert_allclose(B_est[:, 1:], Ju_true, rtol=1e-5, atol=1e-6)
