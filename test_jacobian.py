import numpy as np
from jacobian_tools import evaljacobianf, finite_difference_jacobian, jacobian_testbench
from evalf_bacterial import evalf   # your evalf (now accepts both 1-D and column)
from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference

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
x_flat = np.array([10.0, 5.0, 2.0, 1.0, 0.2])  # 1-D state (n1,n2,n3,R,C)
N = x_flat.size
x_col = x_flat.reshape((N, 1))                 # column-shaped state for external FD

u = np.array([0.5, 0.1])

# Analytic Jacobian: give evaljacobianf the 1-D x (it expects a 1-D array)
J_analytic = evaljacobianf(x_flat, p, u)

# Finite-difference Jacobian: the external function expects a column (N,1)
# It returns (J_fd, dxFD) according to the external implementation.
J_fd_result = eval_Jf_FiniteDifference(evalf, x_col, p, u)

# normalize return (handle either J or (J,dx) just in case)
if isinstance(J_fd_result, tuple) and len(J_fd_result) == 2:
    J_fd, dxFD = J_fd_result
else:
    J_fd = J_fd_result
    dxFD = None

J_fd = np.asarray(J_fd, dtype=float)   # ensure numpy array

print("Used dxFD =", dxFD)

# compare norms (ensure shapes match)
if J_fd.shape != J_analytic.shape:
    raise RuntimeError(f"Shape mismatch: J_fd.shape={J_fd.shape}, J_analytic.shape={J_analytic.shape}")

diff_fro = np.linalg.norm(J_fd - J_analytic, ord='fro')
rel_err = diff_fro / (np.linalg.norm(J_analytic, ord='fro') + 1e-30)
print("Frobenius norm difference (analytic vs FD):", diff_fro)
print("Relative Frobenius error:", rel_err)

# run the testbench sweep of dx factors and print results
# jacobian_testbench uses evalf and evaljacobianf internally; pass the 1-D x
results = jacobian_testbench(evalf, x_flat, p, u)
for fac, err in results:
    print(f"dx_factor={fac:.1e}, frobenius_error={err:.3e}")
