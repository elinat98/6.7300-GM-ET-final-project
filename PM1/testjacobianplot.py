#!/usr/bin/env python3
"""
Jacobian dx-sweep using external eval_Jf_FiniteDifference (scalar forward-diff).
Saves a CSV and a log-log plot of Frobenius error vs absolute dx scalar.

Place next to:
  - evalf_bacterial.py      (must accept column (N,1) and 1-D inputs)
  - jacobian_tools.py
  - eval_Jf_FiniteDifference.py  (external FD function that expects column x)
Run:
  python test_jacobian_sweep_plot_externalFD.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import warnings
from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from tools.eval_Jf_FiniteDifference import eval_Jf_FiniteDifference as external_evalJ

# model / point to test
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

x_flat = np.asarray([10.0, 5.0, 2.0, 1.0, 0.2])   # 1-D preferred for analytic jacobian
N = x_flat.size
x_col = x_flat.reshape((N, 1))                    # column vector for external FD
u = np.array([0.5, 0.1])

# analytic Jacobian (make sure it's a numeric numpy array)
J_analytic = np.asarray(evaljacobianf(x_flat, p, u), dtype=float)
if J_analytic.shape != (N, N):
    raise RuntimeError(f"Analytic Jacobian has unexpected shape {J_analytic.shape}, expected {(N,N)}")

# base scalar dx consistent with external function's heuristic:
eps = np.finfo(float).eps
base_dx_scalar = 2.0 * np.sqrt(eps) * np.sqrt(1.0 + np.linalg.norm(x_flat, np.inf))
print(f"Base scalar dx (external FD heuristic) = {base_dx_scalar:.3e}")

# factors to sweep — up to 1e5 down to 1e-12 (you can change these)
dx_factors = np.logspace(5, -12, num=100)

# storage
dx_scalars = []
errors = []

# run sweep: set p_temp['dxFD'] scalar and call external_evalJ(evalf, x_col, p_temp, u)
for fac in dx_factors:
    dx_scalar = base_dx_scalar * fac
    p_temp = dict(p)  # shallow copy of p
    p_temp['dxFD'] = float(dx_scalar)   # external expects scalar p.dxFD
    # call external function (it may return J or (J, dx) per variant)
    result = external_evalJ(evalf, x_col, p_temp, u)
    if isinstance(result, tuple) and len(result) == 2:
        J_fd, used_dx = result
    else:
        J_fd = result
        used_dx = p_temp['dxFD']   # fallback, should normally not happen
    J_fd = np.asarray(J_fd, dtype=float)
    # ensure shapes
    if J_fd.shape != (N, N):
        raise RuntimeError(f"External FD returned shape {J_fd.shape} (expected {(N,N)}) for factor {fac:.1e}. "
                           f"Check evalf's return shape when called with column inputs.")
    # compute Frobenius error
    diff_norm = np.linalg.norm(J_fd - J_analytic, ord='fro')
    dx_scalars.append(float(used_dx))
    errors.append(diff_norm)

# convert to arrays
dx_scalars = np.array(dx_scalars)
errors = np.array(errors)

# save CSV
out_csv = Path("jacobian_error_table_externalFD.csv")
with out_csv.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["factor", "dx_scalar", "frobenius_error"])
    for fac, dxs, err in zip(dx_factors, dx_scalars, errors):
        writer.writerow([f"{fac:.5e}", f"{dxs:.5e}", f"{err:.5e}"])
print("Saved CSV:", out_csv.resolve())

# find minimum error
imin = np.argmin(errors)
best_fac = dx_factors[imin]
best_dx = dx_scalars[imin]
best_err = errors[imin]
print(f"Minimum Frobenius error {best_err:.3e} at dx_factor = {best_fac:.3e}, dx = {best_dx:.3e}")

# Plot error vs absolute scalar dx
fig, ax = plt.subplots(figsize=(8,5))
ax.loglog(dx_scalars, errors, marker='o', linestyle='-')
ax.set_xlabel('Absolute scalar dx used by external FD')
ax.set_ylabel('Frobenius norm error ||J_fd - J_analytic||_F')
ax.set_title('Jacobian FD error vs absolute scalar dx (external eval_Jf_FiniteDifference)')
ax.grid(True, which='both', ls='--')

# mark minimum and annotate in top-right
ax.plot([best_dx], [best_err], marker='s', markersize=8, color='tab:red')
ax.annotate(
    f"min err={best_err:.2e}\nfac={best_fac:.2e}\ndx={best_dx:.2e}",
    xy=(best_dx, best_err),
    xycoords='data',
    xytext=(0.98, 0.98),
    textcoords='axes fraction',
    ha='right', va='top',
    fontsize=9,
    arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.9)
)

out_png = Path('jacobian_error_vs_dx_scalar_externalFD.png')
fig.tight_layout()
fig.savefig(out_png, bbox_inches='tight')
plt.close(fig)
print("Saved plot to:", out_png.resolve())
