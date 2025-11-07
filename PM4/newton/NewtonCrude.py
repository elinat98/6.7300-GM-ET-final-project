import numpy as np
import time


def newton_crude(eval_f,
                 x0,
                 p,
                 u,
                 jacobian_fn,
                 tol_f=1e-8,
                 tol_dx=1e-8,
                 max_iter=50,
                 verbose=True):
    """
    Simple multi-dimensional Newton method without damping/line search:
      J(x_k) * Delta = -f(x_k)
      x_{k+1} = x_k + Delta
    Stops when ||f||_2 <= tol_f or ||Delta||_2 <= tol_dx, or max_iter reached.
    """
    def _col(z):
        z = np.asarray(z, dtype=float)
        return z.reshape((-1, 1)) if z.ndim == 1 else z

    def _norm2(v):
        return float(np.linalg.norm(np.asarray(v, dtype=float)))

    x = _col(x0).copy()
    N = x.shape[0]
    t0 = time.perf_counter()

    f = _col(eval_f(x, p, u))
    fn = _norm2(f)
    history = [(0, fn)]

    converged = False
    for k in range(1, max_iter + 1):
        if fn <= tol_f:
            converged = True
            break
        J = np.asarray(jacobian_fn(x.ravel(), p, u), dtype=float)
        try:
            Delta = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            Delta, *_ = np.linalg.lstsq(J, -f, rcond=None)
        dxn = _norm2(Delta)
        x = x + Delta
        f = _col(eval_f(x, p, u))
        fn = _norm2(f)
        history.append((k, fn))
        if verbose:
            print(f"iter {k:02d}: ||f||={fn:.3e}, ||dx||={dxn:.3e}")
        if dxn <= tol_dx:
            converged = True
            break

    t1 = time.perf_counter()
    approx_mem_bytes = int(8 * (N * N + 6 * N))  # dense J + vectors
    info = {
        'converged': converged,
        'iters': len(history) - 1,
        'residual_norm': fn,
        'history': history,
        'time_s': t1 - t0,
        'approx_mem_bytes': approx_mem_bytes
    }
    return x, info


