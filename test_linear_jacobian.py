# test_linear_jacobian.py
import numpy as np
from jacobian_tools import evaljacobianf, finite_difference_jacobian
from evalf_bacterial import evalf

m = 3
# Linearizing choices: K=0 => monod=1 (for R>0), set C=0 => hill=1
p = {
    'Q': np.eye(m),
    'rmax': np.array([1.2, 0.8, 0.5]),  # constants
    'K': np.zeros(m),                    # K=0 -> monod=1 (if R>0)
    'alpha': np.zeros(m),                # optionally set alpha=0 to simplify
    'd0': np.array([0.3, 0.2, 0.1]),
    'IC50': np.array([1.0, 1.0, 1.0]),
    'h': np.ones(m),
    'kC': 0.05
}
# pick x with R>0 and C=0
x = np.array([5.0, 3.0, 1.0, 1.0, 0.0])  # n1,n2,n3,R,C

# Analytic Jacobian
J_analytic = evaljacobianf(x, p, np.array([0.0, 0.0]))
print("Analytic J:\n", J_analytic)

# Finite-difference Jacobian (central, per-component scaled)
J_fd = finite_difference_jacobian(evalf, x, p, np.array([0.0, 0.0]), dx_option='scaled', method='central')
print("FD J:\n", J_fd)

# Compare
diff = J_fd - J_analytic
fro = np.linalg.norm(diff, ord='fro')
rel = fro / (np.linalg.norm(J_analytic, ord='fro') + 1e-30)
print(f"Frobenius |J_fd - J_analytic|_F = {fro:.3e}, relative = {rel:.3e}")

# Print max absolute difference entry and its indices
i,j = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
print("Max abs diff at", (i,j), "value", diff[i,j])
