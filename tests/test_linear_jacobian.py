# tests/test_linear_jacobian.py
import numpy as np
from jacobian_tools import evaljacobianf
from evalf_bacterial import evalf
from tools.eval_Jf_FiniteDifference import eval_Jf_FiniteDifference


def test_linear_jacobian_matches_external_fd():
    m = 3
    p = {
        'Q': np.eye(m),
        'rmax': np.array([1.2, 0.8, 0.5]),
        'K': np.zeros(m),    # linearizing choice
        'alpha': np.zeros(m),
        'd0': np.array([0.3, 0.2, 0.1]),
        'IC50': np.array([1.0, 1.0, 1.0]),
        'h': np.ones(m),
        'kC': 0.05
    }
    x_flat = np.array([5.0, 3.0, 1.0, 1.0, 0.0])   # 1-D state
    u = np.array([0.0, 0.0])

    # analytic Jacobian (expects 1-D x)
    J_analytic = np.asarray(evaljacobianf(x_flat, p, u), dtype=float)

    # external FD expects a column-shaped x (N,1)
    N = x_flat.size
    x_col = x_flat.reshape((N, 1))

    result = eval_Jf_FiniteDifference(evalf, x_col, p, u)

    # normalize return (handle either J or (J, dx) just in case)
    if isinstance(result, tuple) and len(result) == 2:
        J_fd, dxFD = result
    else:
        J_fd = result
        dxFD = None

    J_fd = np.asarray(J_fd, dtype=float)

    # shape sanity check
    assert J_fd.shape == J_analytic.shape, f"Shape mismatch: J_fd={J_fd.shape}, J_analytic={J_analytic.shape}"

    # Frobenius norm difference
    diff = np.linalg.norm(J_fd - J_analytic, ord='fro')
    assert diff < 1e-6, f"Jacobian mismatch: Fro={diff}"
