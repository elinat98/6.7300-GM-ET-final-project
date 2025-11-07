import numpy as np
import time


def newton_nd_gcr(eval_f,
                  x0,
                  p,
                  u,
                  tol=1e-8,
                  max_outer=30,
                  gcr_tol=1e-2,
                  gcr_max_iter=50,
                  fd_eps=1e-7,
                  line_search=True,
                  ls_alpha=1e-4,
                  ls_beta=0.5,
                  verbose=True):
    """
    Jacobian-free Newton with GCR (matrix-free) linear solver.
    Solves f(x,p,u)=0 using outer Newton iterations and inner GCR on
    J(x_k) * delta = -f(x_k), using Jv ≈ (f(x+eps*v)-f(x))/eps.

    Returns x (column) and info dict with histories and cost counters.
    """
    def _col(z):
        z = np.asarray(z, dtype=float)
        return z.reshape((-1, 1)) if z.ndim == 1 else z

    def _norm(v):
        return float(np.linalg.norm(np.asarray(v, dtype=float).ravel()))

    # Count function evaluations
    f_evals = 0
    def f_eval_counted(xc):
        nonlocal f_evals
        f_evals += 1
        return _col(eval_f(xc, p, u))

    x = _col(x0).copy()
    N = x.shape[0]
    t_start = time.perf_counter()
    f = f_eval_counted(x)
    res0 = _norm(f)
    history = [(0, res0)]
    outer_lin_iters = []
    peak_k_global = 0
    cum_f_evals = [f_evals]

    if verbose:
        print(f"outer 00: ||f||={res0:.3e}")

    for k in range(1, max_outer + 1):
        if _norm(f) <= tol:
            break

        # Define Jv operator at current x, using finite-difference
        x_norm = _norm(x)
        def Jv(v):
            v = _col(v)
            vnorm = _norm(v)
            if vnorm == 0.0:
                return np.zeros_like(v)
            eps = fd_eps * (1.0 + x_norm) / vnorm
            return (f_eval_counted(x + eps * v) - f) / eps

        # GCR solve: A s = -f, A=J(x)
        b = -f
        s = np.zeros_like(x)
        r = b.copy()
        V = []  # list of v_i = A p_i
        P = []  # list of p_i
        lin_iter = 0
        bnorm = max(_norm(b), 1e-16)
        while _norm(r) > gcr_tol * bnorm and lin_iter < gcr_max_iter:
            p_i = r.copy()
            v_i = Jv(p_i)
            # A-orthogonalize against previous directions
            for (vj, pj) in zip(V, P):
                beta = float(np.dot(v_i.ravel(), vj.ravel()) / max(1e-30, np.dot(vj.ravel(), vj.ravel())))
                v_i = v_i - beta * vj
                p_i = p_i - beta * pj
            v_norm2 = float(np.dot(v_i.ravel(), v_i.ravel()))
            if v_norm2 <= 1e-30:
                break  # stagnation
            alpha = float(np.dot(v_i.ravel(), r.ravel()) / v_norm2)
            s = s + alpha * p_i
            r = r - alpha * v_i
            V.append(v_i)
            P.append(p_i)
            lin_iter += 1
            if lin_iter > peak_k_global:
                peak_k_global = lin_iter

        outer_lin_iters.append(lin_iter)

        # Line search on ||f||^2
        step = 1.0
        if line_search:
            f0_norm2 = float(np.dot(f.ravel(), f.ravel()))
            while step > 1e-8:
                x_trial = x + step * s
                f_trial = f_eval_counted(x_trial)
                if float(np.dot(f_trial.ravel(), f_trial.ravel())) <= (1 - ls_alpha * step) * f0_norm2:
                    x = x_trial
                    f = f_trial
                    break
                step *= ls_beta
            else:
                # accept tiny step
                x = x + step * s
                f = f_eval_counted(x)
        else:
            x = x + s
            f = f_eval_counted(x)

        rk = _norm(f)
        history.append((k, rk))
        cum_f_evals.append(f_evals)
        if verbose:
            print(f"outer {k:02d}: ||f||={rk:.3e} (lin iters {lin_iter}, f_evals {f_evals})")
        if rk <= tol:
            break

    t_end = time.perf_counter()
    # crude peak memory model (bytes): store up to peak_k_global pairs of N-vectors plus a few work vectors
    approx_mem_bytes = int(8 * (N * (2 * max(1, peak_k_global) + 8)))
    info = {
        'converged': _norm(f) <= tol,
        'iters': len(history) - 1,
        'residual_norm': _norm(f),
        'history': history,
        'outer_lin_iters': outer_lin_iters,
        'f_evals': f_evals,
        'cum_f_evals': cum_f_evals,
        'peak_k': peak_k_global,
        'time_s': t_end - t_start,
        'approx_mem_bytes': approx_mem_bytes
    }
    return x, info


