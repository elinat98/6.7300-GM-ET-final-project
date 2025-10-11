# linearize.py
"""
Linearization helper for f(x,p,u).

Main function:
    A, Ju, K0, B = linearize_f(evalf, evaljacobianf=None, x0, p, u0, du=None, fd_method='central')

- evalf(x, p, u) should return f(x,p,u). Accepts either:
    * 1-D x (shape (N,)) -> evalf returns 1-D shape (N,)
    * column x (shape (N,1)) -> evalf may return column (N,1) or 1-D
  The function below normalizes shapes so you can pass either.

- If evaljacobianf is provided it will be used to compute A = df/dx.
  If None, a finite-difference jacobian is computed (calls jacobian_tools.finite_difference_jacobian).

Returns:
  A   : (N,N) df/dx
  Ju  : (N, m_u) df/du
  K0  : (N,) constant term defined by K0 = f0 - A @ x0 - Ju @ u0
  B   : (N, m_u+1) combined [K0, Ju]
"""

from typing import Callable, Optional, Tuple
import numpy as np

def _as_1d_or_col(x):
    """Return (flat1d, was_column_bool)"""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel(), True
    else:
        return arr.ravel(), False

def _normalize_f_output(fout):
    """Return 1-D numpy array from evalf output which may be (N,) or (N,1)"""
    arr = np.asarray(fout, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    return arr.ravel()

def linearize_f(
    evalf: Callable,
    evaljacobianf: Optional[Callable],
    x0,
    p,
    u0,
    du: Optional[float or np.ndarray] = None,
    fd_method: str = 'central'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearize f about (x0, u0).

    Parameters
    ----------
    evalf : callable(x, p, u) -> f
    evaljacobianf : callable(x, p, u) -> Jf or None
        If None -> compute finite-difference Jacobian using jacobian_tools.finite_difference_jacobian.
    x0 : array-like (N,) or (N,1)
    p : parameter dict (passed through to evalf and evaljacobianf)
    u0 : array-like (m_u,)
    du : scalar or array-like (m_u,) perturbation for finite-difference on inputs (optional)
    fd_method : 'central' or 'forward'

    Returns
    -------
    A : (N,N) df/dx
    Ju: (N,m_u) df/du
    K0: (N,) constant vector with K0 = f(x0,u0) - A x0 - Ju u0
    B : (N, m_u+1) concatenation [K0 (col), Ju]
    """
    # normalize x0 and detect column
    x0_flat, was_col = _as_1d_or_col(x0)
    x0 = x0_flat.copy()
    u0 = np.asarray(u0, dtype=float).ravel()

    N = x0.size
    m_u = u0.size

    # evaluate f0
    f0 = _normalize_f_output(evalf(x0 if not was_col else x0.reshape((N,1)), p, u0))

    # A: analytic or FD
    if evaljacobianf is not None:
        A = np.asarray(evaljacobianf(x0, p, u0), dtype=float)
        if A.shape != (N, N):
            raise ValueError(f"evaljacobianf returned shape {A.shape}, expected {(N,N)}")
    else:
        # lazy import to avoid a hard dependency if user supplies analytic function
        try:
            from jacobian_tools import finite_difference_jacobian
        except Exception as e:
            raise ImportError("No analytic jacobian supplied and unable to import jacobian_tools.finite_difference_jacobian") from e
        # finite_difference_jacobian expects evalf signature f(x,p,u)
        A = np.asarray(finite_difference_jacobian(evalf, x0, p, u0, dx_option='scaled', method='central'), dtype=float)
        if A.shape != (N, N):
            # finite_difference_jacobian returns (M,N) matrix; enforce square
            raise ValueError(f"finite_difference_jacobian returned shape {A.shape}, expected {(N,N)}")

    # Ju: compute by finite-difference in u
    eps = np.finfo(float).eps
    if du is None:
        # per-component sensible choice (similar to jacobian_tools scaled)
        du_vec = 2.0 * np.sqrt(eps) * np.maximum(1.0, np.abs(u0))
    else:
        du_arr = np.asarray(du, dtype=float).ravel()
        if du_arr.size == 1:
            du_vec = np.repeat(float(du_arr), m_u)
        elif du_arr.size == m_u:
            du_vec = du_arr
        else:
            raise ValueError("du must be scalar or length equal to len(u0)")

    Ju = np.zeros((N, m_u), dtype=float)
    for j in range(m_u):
        delta = float(du_vec[j])
        if delta <= 0:
            raise ValueError("du perturbations must be positive")
        ej = np.zeros(m_u); ej[j] = 1.0
        if fd_method == 'central':
            up = u0 + delta * ej
            um = u0 - delta * ej
            fp = _normalize_f_output(evalf(x0 if not was_col else x0.reshape((N,1)), p, up))
            fm = _normalize_f_output(evalf(x0 if not was_col else x0.reshape((N,1)), p, um))
            Ju[:, j] = (fp - fm) / (2.0 * delta)
        elif fd_method == 'forward':
            up = u0 + delta * ej
            fp = _normalize_f_output(evalf(x0 if not was_col else x0.reshape((N,1)), p, up))
            Ju[:, j] = (fp - f0) / delta
        else:
            raise ValueError("fd_method must be 'central' or 'forward'")

    # K0
    K0 = f0 - A.dot(x0) - Ju.dot(u0)

    # B as [K0_col, Ju]
    B = np.concatenate([K0.reshape((N,1)), Ju], axis=1)

    return A, Ju, K0, B
