import numpy as np
from jacobian_tools import evaljacobianf, finite_difference_jacobian, jacobian_testbench
from evalf_bacterial import evalf   # your evalf from Part C

# --- small example problem ---
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
x = np.array([10.0, 5.0, 2.0, 1.0, 0.2])  # n1,n2,n3,R,C
u = np.array([0.5, 0.1])

# analytic Jacobian
J_analytic = evaljacobianf(x, p, u)

# finite-diff Jacobian (central, scaled dx)
J_fd = finite_difference_jacobian(evalf, x, p, u, dx_option='scaled', method='central')

# compare norms
diff_fro = np.linalg.norm(J_fd - J_analytic, ord='fro')
print("Frobenius norm difference (analytic vs FD):", diff_fro)

# run the testbench sweep of dx factors and print results
results = jacobian_testbench(evalf, x, p, u)
for fac, err in results:
    print(f"dx_factor={fac:.1e}, frobenius_error={err:.3e}")
