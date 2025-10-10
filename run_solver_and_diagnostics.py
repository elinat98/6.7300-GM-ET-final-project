# run_solver_and_diagnostics.py
import numpy as np
import matplotlib.pyplot as plt
from jacobian_tools import evaljacobianf
from evalf_bacterial import evalf
from tools.SimpleSolver import SimpleSolver   # ensure this import matches your file

# model params (example)
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

# initial condition (column) required by SimpleSolver
x0 = np.array([10.0, 5.0, 2.0, 1.0, 0.2]).reshape((-1,1))

# supply u as function (SimpleSolver likely expects this)
def eval_u(t):
    return np.array([0.5, 0.1])

# Run solver (choose NumIter/w suitable for you)
NumIter = 200
w = 0.01   # initial value
X, t = SimpleSolver(evalf, x0, p, eval_u, NumIter=NumIter, w=w)

print("X shape:", X.shape, "t len:", len(t))
final = X[:, -1].ravel()
print("Final state:", final)

# Quick checks
m = X.shape[0] - 2
n_final = final[:m]
R_final = final[m]
C_final = final[m+1]
print("Total N:", n_final.sum())
print("Fractions:", n_final / n_final.sum())

# Compute Jacobian and eigenvalues at final state (use 1-D x for evaljacobianf)
from numpy.linalg import eigvals
J = evaljacobianf(final, p, eval_u(t[-1]))
eig = eigvals(J)
print("Eigenvalues:", eig)
print("Real parts:", np.real(eig))

# Plot time series
plt.figure(figsize=(9,6))
for i in range(m):
    plt.plot(t, X[i,:].ravel(), label=f"n{i+1}")
plt.plot(t, X[m,:].ravel(), label='R', linestyle='--')
plt.plot(t, X[m+1,:].ravel(), label='C', linestyle=':')
plt.legend()
plt.xlabel('time')
plt.ylabel('state')
plt.title(f'Simulation (w={w})')
plt.grid(True)
plt.show()
