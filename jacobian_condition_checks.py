# jacobian_condition_checks.py
import numpy as np
from jacobian_tools import evaljacobianf
from evalf_bacterial import evalf
from tools.SimpleSolver import SimpleSolver
import matplotlib.pyplot as plt
from numpy.linalg import svd, cond, eigvals

# Setup model & simulate (reuse previous settings)
m = 3
p = { 'Q': np.eye(m), 'rmax': np.array([1.0,0.8,0.6]), 'K': np.array([0.5,0.4,0.3]),
      'alpha': np.array([0.1,0.1,0.1]), 'd0': np.array([0.2,0.15,0.1]),
      'IC50': np.array([1.0,0.8,1.2]), 'h': np.array([1.0,1.0,1.0]), 'kC': 0.05 }

x0 = np.array([10.0,5.0,2.0,1.0,0.2]).reshape((-1,1))
def ufun(t): return np.array([0.5,0.1])

X, t = SimpleSolver(evalf, x0, p, ufun, NumIter=400, w=0.01)

N = X.shape[0]
conds = []
min_svs = []
for k in range(X.shape[1]):
    xk = X[:, k].ravel()
    J = evaljacobianf(xk, p, ufun(t[k]))
    # compute singular values
    s = np.linalg.svd(J, compute_uv=False)
    cond_num = s[0] / (s[-1] + 1e-30)
    conds.append(cond_num)
    min_svs.append(s[-1])

conds = np.array(conds)
min_svs = np.array(min_svs)

print("Cond stats: min, median, max:", np.min(conds), np.median(conds), np.max(conds))
print("Min singular value stats (min, median, max):", np.min(min_svs), np.median(min_svs), np.max(min_svs))

# Plot condition number over time
plt.figure(figsize=(8,4))
plt.semilogy(t, conds, label='cond(J)')
plt.xlabel('time'); plt.ylabel('cond(J)'); plt.grid(True)
plt.title('Jacobian condition number over trajectory')
plt.show()

# Grid scan over R and C (n fixed at some values) to find near singular spots
n_fixed = np.array([5.0, 3.0, 1.0])
R_vals = np.linspace(0.01, 2.0, 50)
C_vals = np.linspace(0.0, 1.0, 50)
cond_grid = np.zeros((R_vals.size, C_vals.size))
minsv_grid = np.zeros_like(cond_grid)

for i,Rv in enumerate(R_vals):
    for j,Cv in enumerate(C_vals):
        x_try = np.concatenate([n_fixed, np.array([Rv, Cv])])
        J = evaljacobianf(x_try, p, np.array([0.5,0.1]))
        s = np.linalg.svd(J, compute_uv=False)
        cond_grid[i,j] = s[0] / (s[-1] + 1e-30)
        minsv_grid[i,j] = s[-1]

# Find location of minimum singular value (closest to singular)
imin, jmin = np.unravel_index(np.argmin(minsv_grid), minsv_grid.shape)
print("Min singular value at R,C:", R_vals[imin], C_vals[jmin], "min sv:", minsv_grid[imin,jmin])
plt.figure(figsize=(8,4))
plt.contourf(C_vals, R_vals, np.log10(minsv_grid + 1e-30), levels=40, cmap='viridis')
plt.colorbar(label='log10(min singular value)')
plt.xlabel('C'); plt.ylabel('R'); plt.title('min singular value (log10) across R,C')
plt.show()
