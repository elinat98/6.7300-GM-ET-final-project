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
import matplotlib.pyplot as plt
import numpy as np


def f_scalar(x, p, u):
    # f(x) = sin(x) -> roots at k*pi. Accepts (1,) or (1,1), returns matching shape.
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        val = np.sin(x[0, 0])
        return np.array([[val]])
    else:
        return np.array([np.sin(x[0])])


def J_scalar(x, p, u):
    # J = [cos(x)] (1x1)
    xval = float(np.asarray(x).ravel()[0])
    return np.array([[np.cos(xval)]], dtype=float)


def run_case(x0, label):
    x0_col = np.array([x0]).reshape((1, 1))
    x_star, info = newton_solve(f_scalar, x0_col, p={}, u=np.array([]), jacobian_fn=J_scalar, max_iter=25, tol=1e-12, verbose=True)
    print(f"\nCase: {label}")
    print("  Converged:", info['converged'])
    print("  Iters:", info['iters'])
    print("  x*:", x_star.ravel())
    print("  f(x*):", f_scalar(x_star, {}, np.array([])).ravel())
    # Compare JFNK-GCR from same initial
    fej = {'n': 0}
    def f_counted(x, p, u):
        fej['n'] += 1
        return f_scalar(x, p, u)
    _xj, info_j = newton_nd_gcr(f_counted, x0_col, {}, np.array([]), tol=1e-12, max_outer=25, gcr_tol=1e-2, gcr_max_iter=25, fd_eps=1e-7, line_search=True, verbose=False)
    # Save per-case cost bar
    fen = {'n': 0}
    def f_counted2(x, p, u):
        fen['n'] += 1
        return f_scalar(x, p, u)
    _xn, info_n = newton_crude(f_counted2, x0_col, {}, np.array([]), jacobian_fn=J_scalar, tol_f=1e-12, tol_dx=1e-12, max_iter=25, verbose=False)
    # Also backtracking Newton
    fel = {'n': 0}
    def f_counted_ls(x, p, u):
        fel['n'] += 1
        return f_scalar(x, p, u)
    _xls, info_ls = newton_solve(f_counted_ls, x0_col, {}, np.array([]), jacobian_fn=J_scalar, max_iter=25, tol=1e-12, verbose=False)
    # Homotopy (LS inner)
    feh = {'n': 0}
    def f_counted_h(x, p, u):
        feh['n'] += 1
        return f_scalar(x, p, u)
    _xh, info_h = homotopy_solve(f_counted_h, x0_col, {}, np.array([]), jacobian_fn=J_scalar, lam_steps=8, inner_method="newton_ls", tol=1e-12, verbose=False)
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
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(['Newton (crude)','Newton (LS)','JFNK-GCR','Homotopy'], [fen['n'], fel['n'], info_j.get('f_evals', 0), feh['n']], color=['tab:blue','tab:orange','tab:green','tab:red'])
    ax.set_ylabel('f evals')
    ax.set_title(f'Cost ({label})')
    ax.set_yscale('log')
    fig.tight_layout()
    # sanitize label for filename
    safe_label = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in label)
    out = Path(__file__).with_name(f"compare_cost_fevals_{safe_label}.png")
    fig.savefig(out, dpi=150)
    # Combined cost panel for this case
    fig2, axs = plt.subplots(1, 3, figsize=(10,3))
    to_mb = lambda b: b / (1024*1024)
    axs[0].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], [fen['n'], fel['n'], info_j.get('f_evals', 0), feh['n']], color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[0].set_title('f evals'); axs[0].set_ylabel('count')
    axs[1].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], [info_n.get('time_s', 0.0), info_ls.get('time_s', 0.0), info_j.get('time_s', 0.0), info_h.get('time_s', 0.0)], color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[1].set_title('time (s)')
    axs[2].bar(['Newton (crude)','Newton (LS)','JFNK','Homotopy'], [to_mb(info_n.get('approx_mem_bytes', 0)), to_mb(info_ls.get('approx_mem_bytes', 0)), to_mb(info_j.get('approx_mem_bytes', 0)), to_mb(info_h.get('approx_mem_bytes', 0))], color=['tab:blue','tab:orange','tab:green','tab:red'])
    axs[2].set_title('memory (MB)')
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    for ax in axs: ax.grid(axis='y', ls='--', alpha=0.3)
    fig2.tight_layout()
    out2 = Path(__file__).with_name(f"compare_costs_{safe_label}.png")
    fig2.savefig(out2, dpi=150)
    # Residual comparison for this case
    residual_sets = [
        (info_n.get('history', []), f'Newton (crude)'),
        (info_ls.get('history', []), f'Newton (LS)'),
        (info_j.get('history', []), f'JFNK'),
        (flatten_homotopy_history(info_h), f'Homotopy')
    ]
    out_res_h = Path(__file__).with_name(f"compare_residual_methods_with_homotopy_{safe_label}.png")
    plot_residual_histories([hist for hist, _ in residual_sets],
                            [lab for _, lab in residual_sets],
                            title=f"Residual convergence ({label})",
                            outfile=str(out_res_h),
                            show=True)
    residual_sets_noh = residual_sets[:3]
    out_res_noh = Path(__file__).with_name(f"compare_residual_methods_no_homotopy_{safe_label}.png")
    plot_residual_histories([hist for hist, _ in residual_sets_noh],
                            [lab for _, lab in residual_sets_noh],
                            title=f"Residual convergence ({label}) without homotopy",
                            outfile=str(out_res_noh),
                            show=True)
    return info['history'], label, info_j['history'], f'JFNK {label}'


def main():
    # Good initial guess near 0 -> expect converge to 0
    h1, l1, j1, jl1 = run_case(0.5, label="near root 0")

    # Initial near pi -> converge to pi (another root)
    h2, l2, j2, jl2 = run_case(3.2, label="near root pi")

    # Poor initial near pi/2 where J ~ 0 -> expect trouble
    h3, l3, j3, jl3 = run_case(1.57079632679, label="near derivative zero (pi/2)")

    out_png = Path(__file__).with_name("residual_history.png")
    plot_residual_histories([h1, h2, h3, j1, j2, j3], [l1, l2, l3, jl1, jl2, jl3], title="Residuals: Newton vs JFNK (scalar sin)", outfile=str(out_png), show=True)


if __name__ == "__main__":
    main()


