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
import matplotlib.pyplot as plt
import numpy as np


def eval_u(_t):  # constant inputs (global so helper can see it)
    return np.array([0.5, 0.1])


def run_case(x_guess, label, use_line_search=True, max_iter=25):
    x_star, info = newton_solve(
        evalf,
        x_guess,
        p,
        eval_u(0.0),
        jacobian_fn=evaljacobianf,
        max_iter=max_iter,
        tol=1e-10,
        line_search=use_line_search,
        verbose=True,
    )
    print(f"\nCase: {label} (line_search={use_line_search})")
    print("  Converged:", info['converged'])
    print("  Iterations:", info['iters'])
    print("  Final residual norm:", info['residual_norm'])
    print("  Final iterate x:", x_star.ravel())
    return info


def main():
    m = 3
    global p
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

    # Deliberately poor initial guesses
    x_bad_large = np.array([1e9, 1e9, 1e9, 1e6, 1e6]).reshape((-1, 1))
    x_bad_signed = np.array([1e8, -1e8, 1e8, -1e6, -1e6]).reshape((-1, 1))

    # Run multiple scenarios: with and without line search
    info_ls = run_case(x_bad_large, "huge positive state", use_line_search=True, max_iter=50)
    info_nols = run_case(x_bad_large, "huge positive state", use_line_search=False, max_iter=25)
    info_signed = run_case(x_bad_signed, "mixed signs (unphysical)", use_line_search=False, max_iter=25)

    # Plot residual history
    out_png = Path(__file__).with_name("residual_history.png")
    histories = []
    labels = []
    if 'history' in info_ls:
        histories.append(info_ls['history']); labels.append("huge pos (line search)")
    if 'history' in info_nols:
        histories.append(info_nols['history']); labels.append("huge pos (no line search)")
    if 'history' in info_signed:
        histories.append(info_signed['history']); labels.append("mixed signs (no line search)")
    plot_residual_histories(histories, labels, title="Newton residuals (challenging inits)", outfile=str(out_png), show=True)

    # Compare against JFNK-GCR cost on same x_bad_large
    fe_j = {'n': 0}
    def evalf_counted_j(x, p_, u_):
        fe_j['n'] += 1
        return evalf(x, p_, u_)
    _xj, info_j = newton_nd_gcr(evalf_counted_j, x_bad_large, p, eval_u(0.0),
                                tol=1e-10, max_outer=50, gcr_tol=1e-2, gcr_max_iter=50,
                                fd_eps=1e-7, line_search=True, verbose=True)
    # Count f-evals for classical Newton (crude) on same init
    fe_n = {'n': 0}
    def evalf_counted_n(x, p_, u_):
        fe_n['n'] += 1
        return evalf(x, p_, u_)
    _xn, info_n = newton_crude(evalf_counted_n, x_bad_large, p, eval_u(0.0), jacobian_fn=evaljacobianf, tol_f=1e-10, tol_dx=1e-10, max_iter=50, verbose=False)
    # Also measure Newton with backtracking
    fe_ls = {'n': 0}
    def evalf_counted_ls(x, p_, u_):
        fe_ls['n'] += 1
        return evalf(x, p_, u_)
    _xls, info_ls2 = newton_solve(evalf_counted_ls, x_bad_large, p, eval_u(0.0), jacobian_fn=evaljacobianf, max_iter=50, tol=1e-10, verbose=False)
    # Homotopy (LS inner)
    fe_h = {'n': 0}
    def evalf_counted_h(x, p_, u_):
        fe_h['n'] += 1
        return evalf(x, p_, u_)
    _xh, info_h = homotopy_solve(evalf_counted_h, x_bad_large, p, eval_u(0.0), jacobian_fn=evaljacobianf, lam_steps=8, inner_method="newton_ls", tol=1e-10, verbose=False)
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
        (info_n.get('history', []), 'Newton (crude)'),
        (info_ls2.get('history', []), 'Newton (LS)'),
        (info_j.get('history', []), 'JFNK-GCR'),
        (flatten_homotopy_history(info_h), 'Homotopy (LS inner)')
    ]
    out_res_h = Path(__file__).with_name("compare_residual_methods_with_homotopy.png")
    plot_residual_histories([hist for hist, _ in residual_sets],
                            [lab for _, lab in residual_sets],
                            title="Residual convergence (bad init)",
                            outfile=str(out_res_h),
                            show=True)
    residual_sets_noh = residual_sets[:3]
    out_res_noh = Path(__file__).with_name("compare_residual_methods_no_homotopy.png")
    plot_residual_histories([hist for hist, _ in residual_sets_noh],
                            [lab for _, lab in residual_sets_noh],
                            title="Residual convergence (bad init, no homotopy)",
                            outfile=str(out_res_noh),
                            show=True)
    # Bar plot
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(['Newton (crude)','Newton (LS)','JFNK-GCR','Homotopy'], [fe_n['n'], fe_ls['n'], info_j.get('f_evals', 0), fe_h['n']], color=['tab:blue','tab:orange','tab:green','tab:red'])
    ax.set_ylabel('total f(x) evaluations')
    ax.set_title('Cost comparison (bad init)')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("compare_cost_fevals.png"), dpi=150)
    # Multi-panel cost
    fig2, axs = plt.subplots(1, 3, figsize=(12,4))
    axs[0].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], [fe_n['n'], fe_ls['n'], info_j.get('f_evals', 0), fe_h['n']], color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[0].set_title('f(x) evals'); axs[0].set_ylabel('count')
    to_mb = lambda b: b / (1024*1024)
    axs[1].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], [info_n.get('time_s', 0.0), info_ls2.get('time_s', 0.0), info_j.get('time_s', 0.0), info_h.get('time_s', 0.0)], color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[1].set_title('wall time (s)')
    axs[2].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], [to_mb(info_n.get('approx_mem_bytes', 0)), to_mb(info_ls2.get('approx_mem_bytes', 0)), to_mb(info_j.get('approx_mem_bytes', 0)), to_mb(info_h.get('approx_mem_bytes', 0))], color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[2].set_title('approx memory (MB)')
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    for ax in axs: ax.grid(axis='y', ls='--', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(Path(__file__).with_name("compare_costs.png"), dpi=150)


if __name__ == "__main__":
    main()


