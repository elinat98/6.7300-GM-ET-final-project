#!/usr/bin/env python3
import sys
from pathlib import Path
# Ensure repo root on sys.path so `import PM4...` works when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from tools.SimpleSolver import SimpleSolver
from PM4.newton.ConvergencePlot import plot_residual_histories
import matplotlib.pyplot as plt
from PM4.newton.NewtonNdGCR import newton_nd_gcr
from PM4.newton.NewtonSolver import newton_solve
from PM4.newton.Homotopy import homotopy_solve


def project_nonnegative(x_col):
    x = np.asarray(x_col, dtype=float).reshape((-1, 1))
    x[x < 0.0] = 0.0
    return x


def newton_projected(eval_f,
                     x0,
                     p,
                     u,
                     jacobian_fn,
                     max_iter=50,
                     tol=1e-8,
                     line_search=True,
                     ls_alpha=1e-4,
                     ls_beta=0.5,
                     verbose=True):
    """
    Projected Newton for bound constraints x >= 0 (componentwise).
    At each step: compute Newton direction at current x, then backtrack with projection
    x_trial = Proj_{x>=0}(x + step*s) and accept if Armijo on ||f||^2 holds.
    """
    # Diagnostics we will record for visualization
    accepted_steps = []
    backtrack_tries = []
    active_counts = []
    used_projection_flags = []
    ls_attempt_residuals = []  # list per-iteration: [(step_try, resid_try), ...]

    def _ensure_column(z):
        z = np.asarray(z, dtype=float)
        return z.reshape((-1, 1)) if z.ndim == 1 else z

    def _norm(v):
        return float(np.linalg.norm(np.asarray(v, dtype=float).ravel()))

    x = project_nonnegative(_ensure_column(x0))
    u = np.asarray(u, dtype=float)
    history = []
    proj_uses = 0

    for k in range(1, max_iter + 1):
        f_val = _ensure_column(eval_f(x, p, u))
        r = _norm(f_val)
        history.append((k, r))
        if verbose:
            print(f"iter {k:02d}: ||f||={r:.3e}")
        if r <= tol:
            return x, {'converged': True, 'iters': k, 'residual_norm': r, 'history': history, 'projections': proj_uses}

        J = np.asarray(jacobian_fn(x.ravel(), p, u), dtype=float)
        try:
            s = np.linalg.solve(J, -f_val)
        except np.linalg.LinAlgError:
            s, *_ = np.linalg.lstsq(J, -f_val, rcond=None)

        # Backtracking with projection
        step = 1.0
        used_proj_this_iter = False
        per_iter_attempts = []
        if line_search:
            f0_norm2 = float(np.dot(f_val.ravel(), f_val.ravel()))
            while step > 1e-8:
                x_trial = x + step * s
                x_trial_proj = project_nonnegative(x_trial)
                used_proj_this_iter = used_proj_this_iter or np.any(x_trial_proj < x_trial - 1e-18)
                f_trial = _ensure_column(eval_f(x_trial_proj, p, u))
                per_iter_attempts.append((step, _norm(f_trial)))
                if float(np.dot(f_trial.ravel(), f_trial.ravel())) <= (1 - ls_alpha * step) * f0_norm2:
                    x = x_trial_proj
                    break
                step *= ls_beta
            else:
                # step underflow, accept projected tiny step to avoid stalling
                x = project_nonnegative(x + step * s)
                per_iter_attempts.append((step, _norm(_ensure_column(eval_f(x, p, u)))))
        else:
            x = project_nonnegative(x + s)
            used_proj_this_iter = used_proj_this_iter or np.any(x < 0.0)
            per_iter_attempts.append((1.0, _norm(_ensure_column(eval_f(x, p, u)))))

        if used_proj_this_iter:
            proj_uses += 1
        # Record diagnostics for this iteration
        accepted_steps.append(step)
        backtrack_tries.append(len(per_iter_attempts))
        used_projection_flags.append(bool(used_proj_this_iter))
        active_counts.append(int(np.count_nonzero(x <= 0.0 + 1e-14)))
        ls_attempt_residuals.append(per_iter_attempts)

    return x, {
        'converged': False,
        'iters': k,
        'residual_norm': r,
        'history': history,
        'projections': proj_uses,
        'accepted_steps': accepted_steps,
        'backtrack_tries': backtrack_tries,
        'active_counts': active_counts,
        'used_projection': used_projection_flags,
        'ls_attempt_residuals': ls_attempt_residuals
    }


def main():
    # Model parameters (3 genotypes)
    m = 3
    p = {
        'Q': np.eye(m),
        'rmax': np.array([1.0, 0.8, 0.6]),
        'K': np.array([0.5, 0.4, 0.3]),
        'alpha': np.array([0.1, 0.1, 0.1]),
        'd0': np.array([0.2, 0.15, 0.1]),
        'IC50': np.array([1.0, 0.8, 1.2]),
        'h': np.array([1.0, 1.0, 1.0]),
        'kC': 0.05
    }

    def eval_u(_t):
        return np.array([0.5, 0.1])

    # Obtain a feasible-but-not-perfect guess:
    # integrate briefly and then perturb negative to trigger projection logic.
    x0 = np.array([10.0, 5.0, 2.0, 1.0, 0.2]).reshape((-1, 1))
    X, t = SimpleSolver(evalf, x0, p, eval_u, NumIter=500, w=2e-3, visualize=False)
    x_guess = X[:, -1].reshape((-1, 1))
    # Push some entries slightly negative
    x_guess[0, 0] -= abs(x_guess[0, 0]) * 0.2 + 1e-2
    x_guess[3, 0] -= abs(x_guess[3, 0]) * 0.5 + 1e-2

    print("Initial guess (with negatives):", x_guess.ravel())
    print("Applying projected Newton (x >= 0)...")
    x_star, info = newton_projected(evalf,
                                    x_guess,
                                    p,
                                    eval_u(0.0),
                                    jacobian_fn=evaljacobianf,
                                    max_iter=60,
                                    tol=1e-10,
                                    line_search=True,
                                    verbose=True)

    print("\nProjected Newton result:")
    print("  Converged   :", info['converged'])
    print("  Iterations  :", info['iters'])
    print("  Projections :", info['projections'])
    print("  Final ||f|| :", info['residual_norm'])
    print("  x* (>=0)    :", x_star.ravel())
    print("  Any negative in x*?:", np.any(x_star < 0.0))
    # Visualization of backtracking and projection behavior
    steps = np.array(info.get('accepted_steps', []), dtype=float)
    tries = np.array(info.get('backtrack_tries', []), dtype=float)
    actives = np.array(info.get('active_counts', []), dtype=float)
    used_proj = np.array(info.get('used_projection', []), dtype=bool)
    iters = np.arange(1, len(steps) + 1, dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    # Panel 1: residual
    axes[0].semilogy([k for (k, _) in info['history']],
                     [max(1e-16, float(r)) for (_, r) in info['history']], '-o', ms=3)
    axes[0].set_ylabel('||f||')
    axes[0].set_title('Projected Newton diagnostics')
    axes[0].grid(True, which='both', ls='--', alpha=0.5)
    # Panel 2: accepted step and number of backtracking tries
    if len(steps) > 0:
        axes[1].plot(iters, steps, '-o', ms=3, label='accepted step')
        axes[1].set_ylabel('step')
        axes[1].grid(True, ls='--', alpha=0.5)
        # keep step axis visible even if constant
        ymin = min(steps.min(), 0.0)
        ymax = max(steps.max(), 1.0)
        if np.isclose(ymin, ymax):
            ymin -= 0.1
            ymax += 0.1
        axes[1].set_ylim(ymin, ymax)
        ax2 = axes[1].twinx()
        ax2.bar(iters, tries if len(tries) == len(iters) else np.ones_like(iters),
                alpha=0.25, color='tab:orange', label='# backtrack tries')
        ax2.set_ylabel('tries')
        # ensure tries axis is visible
        tmax = max(float(np.max(tries)) if tries.size else 1.0, 1.0)
        ax2.set_ylim(0, tmax + 0.5)
        # Legend merge
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        axes[1].text(0.5, 0.5, 'no step diagnostics', transform=axes[1].transAxes,
                     ha='center', va='center')
        axes[1].set_ylabel('step')
        axes[1].grid(True, ls='--', alpha=0.5)
    # Panel 3: active set size; shade iterations where projection was used
    if len(actives) > 0:
        axes[2].plot(iters, actives, '-o', ms=3, label='active (x_i=0)')
        for k, used in zip(iters, used_proj):
            if used:
                axes[2].axvspan(k - 0.5, k + 0.5, color='tab:red', alpha=0.1)
        # ensure visible even if all zeros
        amax = max(float(np.max(actives)), 0.0)
        axes[2].set_ylim(-0.5, max(1.0, amax + 0.5))
        axes[2].legend(loc='best')
    else:
        axes[2].text(0.5, 0.5, 'no active constraints data', transform=axes[2].transAxes,
                     ha='center', va='center')
    axes[2].set_xlabel('iteration')
    axes[2].set_ylabel('# active')
    axes[2].grid(True, ls='--', alpha=0.5)
    out_diag = Path(__file__).with_name("projected_backtracking_diagnostics.png")
    fig.tight_layout()
    fig.savefig(out_diag, dpi=150)
    plt.show()
    # Compare cost with JFNK-GCR on same initial guess (no projection inside JFNK)
    fe_n = {'n': 0}
    def evalf_counted_n(x, p_, u_):
        fe_n['n'] += 1
        return evalf(x, p_, u_)
    # Re-run projected Newton quickly to count f-evals
    _xpn, info_pn = newton_projected(evalf_counted_n, x_guess, p, eval_u(0.0), jacobian_fn=evaljacobianf, max_iter=60, tol=1e-10, line_search=True, verbose=False)
    fe_j = {'n': 0}
    def evalf_counted_j(x, p_, u_):
        fe_j['n'] += 1
        return evalf(x, p_, u_)
    _xj, info_j = newton_nd_gcr(evalf_counted_j, x_guess, p, eval_u(0.0), tol=1e-10, max_outer=60, gcr_tol=1e-2, gcr_max_iter=60, fd_eps=1e-7, line_search=True, verbose=False)
    # Backtracking Newton (unconstrained baseline)
    fe_ls = {'n': 0}
    def evalf_counted_ls(x, p_, u_):
        fe_ls['n'] += 1
        return evalf(x, p_, u_)
    _xls, info_ls = newton_solve(evalf_counted_ls, x_guess, p, eval_u(0.0), jacobian_fn=evaljacobianf, max_iter=60, tol=1e-10, verbose=False)
    # Homotopy (LS inner)
    fe_h = {'n': 0}
    def evalf_counted_h(x, p_, u_):
        fe_h['n'] += 1
        return evalf(x, p_, u_)
    _xh, info_h = homotopy_solve(evalf_counted_h, x_guess, p, eval_u(0.0), jacobian_fn=evaljacobianf, lam_steps=8, inner_method="newton_ls", tol=1e-10, verbose=False)
    def flatten_homotopy_history(hinfo):
        seq = []
        idx = 0
        for _, hist in hinfo.get('history_by_lambda', []):
            for (_, res) in hist:
                seq.append((idx, res))
                idx += 1
        if not seq and hinfo.get('residual_norm') is not None:
            seq.append((0, hinfo['residual_norm']))
        return seq
    residual_sets = [
        (info_pn.get('history', []), 'Projected Newton'),
        (info_ls.get('history', []), 'Newton (LS)'),
        (info_j.get('history', []), 'JFNK-GCR'),
        (flatten_homotopy_history(info_h), 'Homotopy (LS inner)')
    ]
    out_res_h = Path(__file__).with_name("compare_residual_methods_with_homotopy.png")
    plot_residual_histories([hist for hist, _ in residual_sets],
                            [lab for _, lab in residual_sets],
                            title="Residual convergence (projected case)",
                            outfile=str(out_res_h),
                            show=True)
    residual_sets_noh = residual_sets[:3]
    out_res_noh = Path(__file__).with_name("compare_residual_methods_no_homotopy.png")
    plot_residual_histories([hist for hist, _ in residual_sets_noh],
                            [lab for _, lab in residual_sets_noh],
                            title="Residual convergence (projected case, no homotopy)",
                            outfile=str(out_res_noh),
                            show=True)
    figc, axc = plt.subplots(figsize=(5,4))
    vals_fe = [max(fe_n['n'],1e-12), max(fe_ls['n'],1e-12), max(info_j.get('f_evals', 0),1e-12), max(fe_h['n'],1e-12)]
    axc.bar(['Proj-Newton','Newton (LS)','JFNK-GCR','Homotopy'], vals_fe, color=['tab:blue','tab:orange','tab:green','tab:red'])
    axc.set_ylabel('total f(x) evaluations')
    axc.set_title('Cost comparison (projected vs JFNK)')
    axc.set_yscale('log')
    figc.tight_layout()
    figc.savefig(Path(__file__).with_name("compare_cost_fevals.png"), dpi=150)
    # Multi-panel costs
    figc2, axc2 = plt.subplots(1, 3, figsize=(12,4))
    to_mb = lambda b: b / (1024*1024)
    vals0 = [max(fe_n['n'],1e-12), max(fe_ls['n'],1e-12), max(info_j.get('f_evals', 0),1e-12), max(fe_h['n'],1e-12)]
    axc2[0].bar(['Proj-Newton','Newton (LS)','JFNK','Homotopy'], vals0, color=['tab:blue','tab:orange','tab:green','tab:red'])
    axc2[0].set_title('f(x) evals'); axc2[0].set_ylabel('count')
    vals1 = [max(info_pn.get('time_s', 1e-12),1e-12), max(info_ls.get('time_s', 1e-12),1e-12), max(info_j.get('time_s', 1e-12),1e-12), max(info_h.get('time_s', 1e-12),1e-12)]
    axc2[1].bar(['Proj-Newton','Newton (LS)','JFNK','Homotopy'], vals1, color=['tab:blue','tab:orange','tab:green','tab:red'])
    axc2[1].set_title('wall time (s)')
    # memory: proj-newton approx like dense J
    vals2 = [max(to_mb(info_pn.get('approx_mem_bytes', 1) if 'approx_mem_bytes' in info_pn else 1),1e-12), max(to_mb(info_ls.get('approx_mem_bytes', 1)),1e-12), max(to_mb(info_j.get('approx_mem_bytes', 1)),1e-12), max(to_mb(info_h.get('approx_mem_bytes', 1)),1e-12)]
    axc2[2].bar(['Proj-Newton','Newton (LS)','JFNK','Homotopy'], vals2, color=['tab:blue','tab:orange','tab:green','tab:red'])
    axc2[2].set_title('approx memory (MB)')
    for ax in axc2:
        ax.set_yscale('log')
        ax.grid(axis='y', ls='--', alpha=0.3)
    figc2.tight_layout()
    figc2.savefig(Path(__file__).with_name("compare_costs.png"), dpi=150)


if __name__ == "__main__":
    main()


