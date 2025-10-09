# run_simplesolver.py
import numpy as np
from evalf_bacterial import evalf
from SimpleSolver import SimpleSolver   # ensure name matches file/class
import matplotlib.pyplot as plt

# --- problem & params ---
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

# initial condition as column vector (SimpleSolver usually uses column)
x0 = np.array([10.0, 5.0, 2.0, 1.0, 0.2]).reshape((-1,1))

# If SimpleSolver expects a time-varying input function u(t):
def eval_u(t):
    # return array-like [uR, uC]
    return np.array([0.5, 0.0])

# Otherwise, if SimpleSolver expects a constant u array, use:
# u_const = np.array([0.5, 0.0])

# --- run solver ---
# If SimpleSolver wants u as a function, pass eval_u; if it wants array, pass u_const.
# The SimpleSolver signature in your repo was used earlier as:
#   X, t = SimpleSolver(evalf, x0, p, eval_u, NumIter=..., w=...)
# so we follow that.
NumIter = 200
w = 0.01   # time step or similar (use what SimpleSolver expects)

X, t = SimpleSolver(evalf, x0, p, eval_u, NumIter=NumIter, w=w)

# --- postprocess / inspect ---
print("Solver returned X shape:", np.asarray(X).shape)
print("Time vector length:", len(t))
# print final state (column)
print("Final state (last column):")
print(X[:, -1].reshape(-1))

# --- quick summaries for the final state ---
final = X[:, -1].ravel()   # last column
n = final[:3]
R = final[3]
C = final[4]
N_total = n.sum()
fractions = n / N_total

# Shannon entropy (nats)
H = -np.sum([p*np.log(p) for p in fractions if p>0.0])

print("Final totals:")
print(f" n: {n}")
print(f" Total N = {N_total:.5g}")
print(f" Fractions = {fractions}")
print(f" Shannon entropy H = {H:.4g} nats ({H/np.log(2):.4g} bits)")
print(f" Resource R = {R:.6g}")
print(f" Antibiotic C = {C:.6g}")

# --- time series plots ---
fig, axs = plt.subplots(2,1, figsize=(8,8), sharex=True)

time = t  # whatever time vector SimpleSolver returned
# Plot genotype counts
for i in range(3):
    axs[0].plot(time, X[i,:].ravel(), label=f"n{i+1}")
axs[0].set_ylabel("Abundance")
axs[0].legend()
axs[0].grid(True)

# Plot R and C
axs[1].plot(time, X[3,:].ravel(), label="R (resource)")
axs[1].plot(time, X[4,:].ravel(), label="C (antibiotic)")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Concentration")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# --- genotype fractions over time ---
fig2, ax2 = plt.subplots(figsize=(8,4))
for i in range(3):
    ax2.plot(time, X[i,:].ravel() / np.sum(X[:3,:], axis=0), label=f"frac n{i+1}")
ax2.set_xlabel("Time")
ax2.set_ylabel("Genotype fraction")
ax2.legend()
ax2.grid(True)
plt.show()

from jacobian_tools import evaljacobianf
import numpy as np

x_final_flat = final  # it's 1-D already; if column, do .ravel()
J = evaljacobianf(x_final_flat, p, np.array([0.5,0.0]))  # use appropriate input u at final time
eigvals = np.linalg.eigvals(J)
print("Eigenvalues of J at final state:", eigvals)
print("Real parts:", np.real(eigvals))