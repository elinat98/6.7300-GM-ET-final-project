#!/usr/bin/env python3
"""
Improved Jacobian dx-sweep + plot: plot vs absolute dx (geometric mean) and save CSV.

Place next to evalf_bacterial.py and jacobian_tools.py and run:
    python test_jacobian_sweep_plot_improved.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import csv

try:
    from evalf_bacterial import evalf
except Exception as e:
    raise ImportError("Cannot import evalf from evalf_bacterial.py: " + str(e))
try:
    from jacobian_tools import evaljacobianf
except Exception as e:
    raise ImportError("Cannot import evaljacobianf from jacobian_tools.py: " + str(e))

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
x = np.array([10.0, 5.0, 2.0, 1.0, 0.2])
u = np.array([0.5, 0.1])

J_analytic = evaljacobianf(x, p, u)

# base per-component dx (scaled)
eps = np.finfo(float).eps
base_dx = 2.0 * np.sqrt(eps) * np.maximum(1.0, np.abs(x))

# factors: user asked to go up to 1e5 (and down to 1e-12)
dx_factors = np.logspace(5, -12, num=100)   # fine sampling

def finite_diff_jacobian_central(f, x, p, u, dx_vec):
    x = np.asarray(x, dtype=float)
    N = x.size
    f0 = np.asarray(f(x, p, u), dtype=float)
    M = f0.size
    J_fd = np.zeros((M, N), dtype=float)
    for k in range(N):
        ek = np.zeros(N); ek[k] = 1.0
        dxk = dx_vec[k]
        if dxk == 0.0:
            raise ValueError("dx component is zero.")
        if dxk <= np.finfo(float).eps:
            warnings.warn(f"dx[{k}] <= machine epsilon; FD dominated by roundoff", RuntimeWarning)
        fk_plus = np.asarray(f(x + dxk * ek, p, u), dtype=float)
        fk_minus = np.asarray(f(x - dxk * ek, p, u), dtype=float)
        J_fd[:, k] = (fk_plus - fk_minus) / (2.0 * dxk)
    return J_fd

# run sweep, compute errors, and representative dx scalar per factor
errors = []
dx_scalars = []   # geometric mean of dx_vec
for fac in dx_factors:
    dx_vec = base_dx * fac
    # representative scalar: geometric mean of dx_vec (avoid zeros)
    dx_scalar = np.exp(np.mean(np.log(np.abs(dx_vec))))
    dx_scalars.append(dx_scalar)
    J_fd = finite_diff_jacobian_central(evalf, x, p, u, dx_vec)
    err = np.linalg.norm(J_fd - J_analytic, ord='fro')
    errors.append(err)

dx_scalars = np.array(dx_scalars)
errors = np.array(errors)

# Save table for inspection
out_csv = Path('jacobian_error_table.csv')
with out_csv.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['factor','dx_scalar','error'])
    for fac, dxs, err in zip(dx_factors, dx_scalars, errors):
        writer.writerow([f"{fac:.5e}", f"{dxs:.5e}", f"{err:.5e}"])

# find min
imin = np.argmin(errors)
best_fac = dx_factors[imin]
best_dx = dx_scalars[imin]
best_err = errors[imin]
print(f"min error {best_err:.3e} at factor {best_fac:.3e}, dx_scalar {best_dx:.3e}")

# Plot: error vs absolute dx scalar 
fig, ax = plt.subplots(figsize=(8,5))
ax.loglog(dx_scalars, errors, marker='o', linestyle='-', markersize=4)
ax.set_xlabel('Representative absolute dx')
ax.set_ylabel('Frobenius norm error')
ax.set_title('Jacobian FD error vs absolute dx ')
ax.grid(True, which='both', ls='--')

# mark the minimum with a small square marker
ax.plot([best_dx], [best_err], marker='s', markersize=8, color='tab:red')

# place the annotation text in the top-right corner of the axes and draw an arrow to the min point
ax.annotate(
    f"min err={best_err:.2e}\nfac={best_fac:.2e}\ndx={best_dx:.2e}",
    xy=(best_dx, best_err),                # point being annotated (data coords)
    xycoords='data',
    xytext=(0.98, 0.98),                   # target text position in axes fraction coords (top-right)
    textcoords='axes fraction',
    ha='right', va='top',
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.9)
)
# Optionally show vertical lines for base_dx values (for intuition)
mean_base = np.exp(np.mean(np.log(base_dx)))
ax.axvline(mean_base, color='gray', linestyle=':', linewidth=1, label='mean base dx')
ax.legend()

out_png = Path('jacobian_error_vs_dx_scalar.png')
fig.tight_layout()
fig.savefig(out_png, bbox_inches='tight')
plt.close(fig)

print("Saved plot to:", out_png.resolve())
print("Saved table to:", out_csv.resolve())
