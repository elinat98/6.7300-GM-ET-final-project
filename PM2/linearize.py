# linearize.py
"""
Linearization helper for the project.

Provides linearize_f(f_eval, J_eval, x0, p, u0, ...) which computes
A = df/dx (at x0,u0), Ju = df/du (at x0,u0), K0 = f(x0,u0) - A x0 - Ju u0

Returns:
  A : (N,N) numpy array
  B : (N, 1 + m_u) numpy array  # [ K0 | Ju ]
"""

import numpy as np

# try to import external FD Jacobian (user said it's in tools)
try:
    from tools.eval_Jf_FiniteDifference import eval_Jf_FiniteDifference as external_evalJ
except Exception:
    external_evalJ = None


def _ensure_flat_vector(x):
    """Return 1-D numpy array from x which may be (N,) or (N,1)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        return x.ravel()
    return x.ravel()


def linearize_f(f_eval, J_eval, x0, p, u0, du=1e-6, fd_method='central'):
    """
    Linearize f around (x0,u0).

    Parameters
    ----------
    f_eval : callable
        f_eval(x, p, u) returning f(x,p,u). Accepts either 1-D x (N,) or column (N,1).
    J_eval : callable or None
        If callable, should return analytic Jacobian J(x,p,u) (shape (N,N)).
        If None, we will use external finite-difference function in tools (if available).
    x0 : array-like
        bias state (N,) or (N,1)
    p : dict
        parameters passed to f_eval and to external FD
    u0 : array-like
        bias input (m_u,) or (m_u,1)
    du : float or array-like
        finite-difference step for u-derivatives (scalar or per-component).
    fd_method : 'forward'|'central'
        method for u finite-difference (central recommended)

    Returns
    -------
    A : ndarray (N,N)
    B : ndarray (N, 1 + m_u)  # columns: [K0, Ju_col0, Ju_col1, ...]
    """
    # normalize x0 and u0 shapes
    x0_flat = _ensure_flat_vector(x0)
    u0 = np.asarray(u0, dtype=float).ravel()
    N = x0_flat.size
    m_u = u0.size

    # Evaluate f at the bias point
    # Keep return shape consistent: we convert any column to flat 1-D
    f0_raw = f_eval(x0 if (np.asarray(x0).ndim == 2 and np.asarray(x0).shape[1] == 1) else x0_flat, p, u0)
    f0 = np.asarray(f0_raw, dtype=float).ravel()
    if f0.size != N:
        raise ValueError(f"f(x0,p,u0) returned shape {np.shape(f0_raw)} -> flattened {f0.size}, expected {N}")

    # Compute A (df/dx)
    if callable(J_eval):
        # user provided analytic Jacobian
        J_analytic = np.asarray(J_eval(x0_flat, p, u0), dtype=float)
        if J_analytic.shape != (N, N):
            raise ValueError(f"Analytic Jacobian returned shape {J_analytic.shape}, expected {(N,N)}")
        A = J_analytic
    else:
        # use external FD Jacobian if available
        if external_evalJ is None:
            raise RuntimeError("No analytic J_eval provided and external eval_Jf_FiniteDifference not available.")
        # external function expects a column-shaped x (N,1)
        x_col = x0_flat.reshape((N, 1))
        J_res = external_evalJ(f_eval, x_col, p, u0)
        # external function may return (J, dx) or J alone
        if isinstance(J_res, tuple) and len(J_res) == 2:
            J_fd, used_dx = J_res
        else:
            J_fd = J_res
            used_dx = None
        A = np.asarray(J_fd, dtype=float)
        if A.shape != (N, N):
            raise RuntimeError(f"External FD Jacobian returned shape {A.shape}, expected {(N,N)}")

    # Compute Ju = df/du (N x m_u) via finite differences on u
    Ju = np.zeros((N, m_u), dtype=float)
    # allow du to be scalar or vector
    if np.isscalar(du):
        du_vec = np.full(m_u, float(du))
    else:
        du_vec = np.asarray(du, dtype=float).ravel()
        if du_vec.size != m_u:
            raise ValueError("du must be scalar or length m_u vector")

    # pick fd_method
    if fd_method not in ('forward', 'central'):
        raise ValueError("fd_method must be 'forward' or 'central'")

    for j in range(m_u):
        dj = du_vec[j]
        if dj <= 0:
            raise ValueError("du must be positive")
        uj = u0.copy()
        if fd_method == 'forward':
            uj_plus = uj.copy(); uj_plus[j] = uj[j] + dj
            f_plus = np.asarray(f_eval(x0_flat, p, uj_plus), dtype=float).ravel()
            Ju[:, j] = (f_plus - f0) / dj
        else:  # central
            uj_plus = uj.copy(); uj_plus[j] = uj[j] + dj
            uj_minus = uj.copy(); uj_minus[j] = uj[j] - dj
            f_plus = np.asarray(f_eval(x0_flat, p, uj_plus), dtype=float).ravel()
            f_minus = np.asarray(f_eval(x0_flat, p, uj_minus), dtype=float).ravel()
            Ju[:, j] = (f_plus - f_minus) / (2.0 * dj)

    # compute K0 = f0 - A x0 - Ju u0
    K0 = f0 - A.dot(x0_flat) - Ju.dot(u0)

    # Build B = [K0 | Ju] shape (N, 1 + m_u)
    B = np.zeros((N, 1 + m_u), dtype=float)
    B[:, 0] = K0
    if m_u > 0:
        B[:, 1:] = Ju

    return A, B
