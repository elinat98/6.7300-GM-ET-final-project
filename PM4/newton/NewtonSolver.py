import numpy as np
import time

try:
    # Prefer analytic Jacobian if available
    from jacobian_tools import evaljacobianf as analytic_jacobian
except Exception:
    analytic_jacobian = None

try:
    # Fallback finite-difference Jacobian helper (expects column x)
    from tools.eval_Jf_FiniteDifference import eval_Jf_FiniteDifference as fd_jacobian
except Exception:
    fd_jacobian = None


def _ensure_column(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape((-1, 1))
    return x


def _norm(v):
    v = np.asarray(v, dtype=float).ravel()
    return float(np.linalg.norm(v))


def newton_solve(eval_f,
                 x0,
                 p,
                 u,
                 jacobian_fn=None,
                 max_iter=50,
                 tol=1e-8,
                 line_search=True,
                 ls_alpha=1e-4,
                 ls_beta=0.5,
                 verbose=True):
    """
    Solve f(x,p,u)=0 via Newton's method with optional backtracking line search.

    Parameters
    ----------
    eval_f : callable
        Vector field f(x,p,u). Accepts x as (N,) or (N,1). Returns same length.
    x0 : array-like
        Initial guess, shape (N,) or (N,1).
    p : dict
        Model parameters (passed through). If fd_jacobian is used, may include p['dxFD'].
    u : array-like
        Input vector (passed through to f and Jacobian).
    jacobian_fn : callable or None
        If callable, must return J(x,p,u) with shape (N,N) for column or flat x.
        If None, uses analytic Jacobian (if importable) else finite differences.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on residual norm.
    line_search : bool
        If True, use backtracking Armijo line search to reduce residual.
    ls_alpha : float
        Armijo parameter in (0, 0.5].
    ls_beta : float
        Step shrink factor in (0, 1).
    verbose : bool
        Print progress.

    Returns
    -------
    x : ndarray (N,1)
        Final iterate (column vector).
    info : dict
        {'converged': bool, 'iters': k, 'residual_norm': r, 'history': [...]}.
    """

    x = _ensure_column(x0).copy()
    u = np.asarray(u, dtype=float)
    N = x.shape[0]
    t_start = time.perf_counter()

    def _J(x_col):
        if callable(jacobian_fn):
            J = jacobian_fn(x_col.ravel(), p, u)
            return np.asarray(J, dtype=float)
        if analytic_jacobian is not None:
            return np.asarray(analytic_jacobian(x_col.ravel(), p, u), dtype=float)
        if fd_jacobian is not None:
            J, _dx = fd_jacobian(lambda xc, pp, uu: _ensure_column(eval_f(xc, pp, uu)), x_col, p, u)
            return np.asarray(J, dtype=float)
        raise RuntimeError("No Jacobian available: provide jacobian_fn or ensure analytic/fd import works.")

    history = []
    converged = False

    for k in range(1, max_iter + 1):
        f_val = _ensure_column(eval_f(x, p, u))
        r = _norm(f_val)
        history.append((k, r))
        if verbose:
            print(f"iter {k:02d}: ||f||={r:.3e}")
        if r <= tol:
            converged = True
            break

        J = _J(x)
        try:
            # Solve J s = -f
            s = np.linalg.solve(J, -f_val)
        except np.linalg.LinAlgError:
            # Try least squares if singular
            s, *_ = np.linalg.lstsq(J, -f_val, rcond=None)

        # Backtracking line search on residual norm
        step = 1.0
        if line_search:
            f0_norm2 = float(np.dot(f_val.ravel(), f_val.ravel()))
            # predicted decrease proxy: use J*s ~ -f, so gradient along s is -||f||^2
            while step > 1e-8:
                x_trial = x + step * s
                f_trial = _ensure_column(eval_f(x_trial, p, u))
                if float(np.dot(f_trial.ravel(), f_trial.ravel())) <= (1 - ls_alpha * step) * f0_norm2:
                    break
                step *= ls_beta
        x = x + step * s

    t_end = time.perf_counter()
    # crude peak memory model (bytes): dense J (N x N) plus a few vectors
    approx_mem_bytes = int(8 * (N * N + 6 * N))
    info = {
        'converged': converged,
        'iters': k,
        'residual_norm': r if 'r' in locals() else None,
        'history': history,
        'time_s': t_end - t_start,
        'approx_mem_bytes': approx_mem_bytes
    }
    return x, info


