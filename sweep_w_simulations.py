import numpy as np
import matplotlib.pyplot as plt
from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from tools.SimpleSolver import SimpleSolver   # your solver
import math

# model (same as you used)
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
x0 = np.array([10.0, 5.0, 2.0, 1.0, 0.2]).reshape((-1,1))

def eval_u(t): return np.array([0.5, 0.1])

w_values = [1e-1]

final_states = []
eig_reals = []
Ns = []
Rs = []
Cs = []

for w in w_values:
    print(f"running SimpleSolver with w={w} ...")
    X, t = SimpleSolver(evalf, x0, p, eval_u, NumIter=200, w=w)
    final = X[:, -1].ravel()
    N_final = float(final[:m].sum())
    Ns.append(N_final); Rs.append(final[m]); Cs.append(final[m+1])
    final_states.append(final)
    J = evaljacobianf(final, p, eval_u(t[-1]))
    eigs = np.linalg.eigvals(J)
    eig_reals.append(np.real(eigs))
    print(f" w={w} final N={N_final:.4g}, R={final[m]:.4g}, C={final[m+1]:.4g}")
print("\nSummary (w, real parts of eigenvalues):")
for w, er in zip(w_values, eig_reals):
    print(f"{w:.3g} {np.round(er,5)}")

# Convergence diagnostics vs base w (smallest w as reference)
ref = 0  # index of reference (w_values[0] if you want)
for i,w in enumerate(w_values):
    rel = np.linalg.norm(final_states[i] - final_states[ref]) / (np.linalg.norm(final_states[ref]) + 1e-30)
    print(f"Relative change from w={w_values[ref]} to w={w}: {rel:.3e}")

# Plot final N, R, C vs w
plt.figure(figsize=(6,4))
plt.plot(w_values, Ns, '-o', label='N_final')
plt.plot(w_values, Rs, '-s', label='R_final')
plt.plot(w_values, Cs, '-^', label='C_final')
plt.xscale('log')
plt.xlabel('w')
plt.ylabel('final value')
plt.legend()
plt.grid(True)
plt.title('Final values vs w')
plt.show()

# Plot leading eigenvalue real parts vs w
lead = [max(er) for er in eig_reals]    # most positive real part
second = [sorted(er, reverse=True)[1] for er in eig_reals]
plt.figure(figsize=(6,4))
plt.plot(w_values, lead, '-o', label='largest real eig')
plt.plot(w_values, second, '-s', label='2nd largest real eig')
plt.xscale('log')
plt.axhline(0, color='k', lw=0.5)
plt.xlabel('w'); plt.ylabel('eig real part')
plt.legend(); plt.grid(True); plt.title('Leading eigenvalues vs w')
plt.show()
