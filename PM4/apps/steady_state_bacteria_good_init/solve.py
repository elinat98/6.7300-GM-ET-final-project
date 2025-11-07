#!/usr/bin/env python3
import sys
from pathlib import Path
# Ensure repo root on sys.path so `import PM4...` works when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from PM4.newton.NewtonSolver import newton_solve
from PM4.newton.ConvergencePlot import plot_residual_histories
from PM4.newton.NewtonNdGCR import newton_nd_gcr
from PM4.newton.NewtonCrude import newton_crude
from PM4.newton.Homotopy import homotopy_solve
from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from tools.SimpleSolver import SimpleSolver
import matplotlib.pyplot as plt
import numpy as np


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

    # Start from a reasonable state and integrate to near steady-state
    x0 = np.array([10.0, 5.0, 2.0, 1.0, 0.2]).reshape((-1, 1))
    X, t = SimpleSolver(evalf, x0, p, eval_u, NumIter=2000, w=1e-3, visualize=False)
    x_guess = X[:, -1].reshape((-1, 1))

    # Small perturbation to avoid exact steady state
    x_guess = x_guess * (1 + 1e-3 * (2*np.random.rand(x_guess.size, 1) - 1))

    # Newton refinement using analytic Jacobian
    x_star, info = newton_solve(evalf, x_guess, p, eval_u(0.0), jacobian_fn=evaljacobianf, max_iter=50, tol=1e-10, verbose=True)

    print("Converged:", info['converged'])
    print("Iterations:", info['iters'])
    print("Final residual norm:", info['residual_norm'])
    print("Steady state x*:", x_star.ravel())
    # Compare with Jacobian-free Newton (GCR)
    # Wrap evalf to count evaluations
    fe_count = {'n': 0}
    def evalf_counted(x, p_, u_):
        fe_count['n'] += 1
        return evalf(x, p_, u_)
    x_jfnk, info_j = newton_nd_gcr(evalf_counted, x_guess, p, eval_u(0.0),
                                   tol=1e-10, max_outer=50, gcr_tol=1e-2, gcr_max_iter=50,
                                   fd_eps=1e-7, line_search=True, verbose=True)
    print("JFNK f-evals:", info_j.get('f_evals'))
    # Plot comparison: residual vs cumulative f evals
    # Crude Newton (no line search)
    fe2 = {'n': 0}
    def evalf_counted2(x, p_, u_):
        fe2['n'] += 1
        return evalf(x, p_, u_)
    _x2, info2 = newton_crude(evalf_counted2, x_guess, p, eval_u(0.0), jacobian_fn=evaljacobianf, tol_f=1e-10, tol_dx=1e-10, max_iter=50, verbose=False)
    # Newton with backtracking (line search)
    fe_ls = {'n': 0}
    def evalf_counted_ls(x, p_, u_):
        fe_ls['n'] += 1
        return evalf(x, p_, u_)
    _xls, info_ls = newton_solve(evalf_counted_ls, x_guess, p, eval_u(0.0), jacobian_fn=evaljacobianf, max_iter=50, tol=1e-10, verbose=False)
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
        (info['history'], 'Newton (LS)'),
        (info2['history'], 'Newton (crude)'),
        (info_j['history'], 'JFNK-GCR'),
        (flatten_homotopy_history(info_h), 'Homotopy (LS inner)')
    ]
    out_res_h = Path(__file__).with_name("compare_residual_methods_with_homotopy.png")
    plot_residual_histories([hist for hist, _ in residual_sets],
                            [lab for _, lab in residual_sets],
                            title="Residual convergence (good init)",
                            outfile=str(out_res_h),
                            show=True)
    # Also plot without homotopy for clearer comparison
    residual_sets_noh = residual_sets[:3]
    out_res_noh = Path(__file__).with_name("compare_residual_methods_no_homotopy.png")
    plot_residual_histories([hist for hist, _ in residual_sets_noh],
                            [lab for _, lab in residual_sets_noh],
                            title="Residual convergence (good init, no homotopy)",
                            outfile=str(out_res_noh),
                            show=True)
    # Cost bar plot
    fig2, ax2 = plt.subplots(figsize=(5,4))
    vals_fe = [max(fe2['n'], 1e-12), max(fe_ls['n'], 1e-12), max(info_j.get('f_evals', 0), 1e-12), max(fe_h['n'], 1e-12)]
    ax2.bar(['Newton (crude)','Newton (LS)','JFNK-GCR','Homotopy'], vals_fe, color=['tab:blue','tab:orange','tab:green','tab:red'])
    ax2.set_ylabel('total f(x) evaluations')
    ax2.set_title('Cost comparison (good init)')
    ax2.set_yscale('log')
    fig2.tight_layout()
    fig2.savefig(Path(__file__).with_name("compare_cost_fevals.png"), dpi=150)
    # Multi-panel cost: f-evals, time, memory
    fig3, axs = plt.subplots(1, 3, figsize=(12,4))
    axs[0].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], vals_fe, color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[0].set_title('f(x) evals'); axs[0].set_ylabel('count')
    vals_time = [max(info2.get('time_s', 1e-12), 1e-12),
                 max(info_ls.get('time_s', 1e-12), 1e-12),
                 max(info_j.get('time_s', 1e-12), 1e-12),
                 max(info_h.get('time_s', 1e-12), 1e-12)]
    axs[1].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], vals_time, color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[1].set_title('wall time (s)')
    to_mb = lambda b: b / (1024*1024)
    vals_mem = [max(to_mb(info2.get('approx_mem_bytes', 1)), 1e-12),
                max(to_mb(info_ls.get('approx_mem_bytes', 1)), 1e-12),
                max(to_mb(info_j.get('approx_mem_bytes', 1)), 1e-12),
                max(to_mb(info_h.get('approx_mem_bytes', 1)), 1e-12)]
    axs[2].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], vals_mem, color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[2].set_title('approx memory (MB)')
    for ax in axs:
        ax.set_yscale('log')
        ax.grid(axis='y', ls='--', alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(Path(__file__).with_name("compare_costs.png"), dpi=150)


if __name__ == "__main__":
    main()


