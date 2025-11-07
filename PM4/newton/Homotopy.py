import numpy as np
import time
from .NewtonSolver import newton_solve
from .NewtonNdGCR import newton_nd_gcr
from .NewtonCrude import newton_crude


def homotopy_solve(eval_f,
                   x0,
                   p,
                   u,
                   jacobian_fn=None,
                   lam_steps=10,
                   inner_method="newton_ls",  # 'newton_ls' | 'newton_crude' | 'jfnk'
                   tol=1e-8,
                   verbose=True):
    """
    Generic homotopy continuation:
      Solve H(x, λ) = (1-λ)*(x - x_ref) + λ f(x,p,u) = 0
      with λ from 0 -> 1 in lam_steps, using previous step's solution as initial guess.
    Inner method can be: Newton with line search, crude Newton, or JFNK-GCR.

    Returns final x and aggregated info: time_s, f_evals, approx_mem_bytes (peak of inner).
    """
    def _col(z):
        z = np.asarray(z, dtype=float)
        return z.reshape((-1, 1)) if z.ndim == 1 else z

    x_ref = _col(x0).copy()
    x = x_ref.copy()
    N = x.shape[0]
    total_f_evals = 0
    t0 = time.perf_counter()
    peak_mem = 0
    histories = []

    for k in range(lam_steps + 1):
        lam = k / lam_steps
        if verbose:
            print(f"[Homotopy] lambda={lam:.3f}")

        # Define H and J_H at this lambda
        feval_counter = {'n': 0}

        def H(xc, pp, uu):
            feval_counter['n'] += 1
            return (1 - lam) * (_col(xc) - x_ref) + lam * _col(eval_f(xc, pp, uu))

        if jacobian_fn is not None:
            def JH(xc, pp, uu):
                Jf = np.asarray(jacobian_fn(np.asarray(xc).ravel(), pp, uu), dtype=float)
                return (1 - lam) * np.eye(N) + lam * Jf
        else:
            JH = None

        # Choose inner solve
        if inner_method == "newton_ls":
            x, info_in = newton_solve(H, x, p, u, jacobian_fn=JH, max_iter=50, tol=tol, line_search=True, verbose=False)
        elif inner_method == "newton_crude":
            if JH is None:
                raise RuntimeError("newton_crude requires jacobian_fn; provide jacobian_fn to homotopy_solve.")
            x, info_in = newton_crude(H, x, p, u, jacobian_fn=JH, tol_f=tol, tol_dx=tol, max_iter=50, verbose=False)
        elif inner_method == "jfnk":
            # JFNK on H: just pass H as eval_f
            x, info_in = newton_nd_gcr(H, x, p, u, tol=tol, max_outer=50, gcr_tol=1e-2, gcr_max_iter=50, fd_eps=1e-7, line_search=True, verbose=False)
        else:
            raise ValueError("inner_method must be 'newton_ls'|'newton_crude'|'jfnk'")

        total_f_evals += feval_counter['n']
        peak_mem = max(peak_mem, int(info_in.get('approx_mem_bytes', 0)))
        histories.append((lam, info_in.get('history', [])))

    t1 = time.perf_counter()
    info = {
        'converged': True,
        'iters': lam_steps,
        'residual_norm': info_in.get('residual_norm', None),
        'history_by_lambda': histories,
        'f_evals': total_f_evals,
        'time_s': t1 - t0,
        'approx_mem_bytes': peak_mem
    }
    return x, info



