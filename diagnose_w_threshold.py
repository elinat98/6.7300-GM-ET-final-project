# diagnose_w_threshold.py
import numpy as np
from pathlib import Path
from find_omega_limits import run_solver_safe, final_state_from_X
# adapt imports if find_omega_limits is named differently in your copy
from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from tools.SimpleSolver import SimpleSolver

# model (same as used by the script)
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
x0_col = np.array([10.0,5.0,2.0,1.0,0.2]).reshape((-1,1))
u = lambda t: np.array([0.5,0.1])

# test many w values in a tight window around the reported 0.876
ws = np.concatenate([
    np.linspace(0.6, 0.9, 16),
    np.linspace(0.85, 1.2, 19)
])
ws = np.unique(np.sort(ws))

print("Testing w values:", ws)
for w in ws:
    ok, X, t, info = run_solver_safe(evalf, SimpleSolver, x0_col, p, u, NumIter=200, w=w)
    status = "OK" if ok else "FAIL"
    print(f"\n== w = {w:.6f} -> {status}: {info}")
    if X is not None:
        X = np.asarray(X)
        try:
            final = final_state_from_X(X)
            print(" final state (last column):", final)
            print(" max abs:", float(np.max(np.abs(X))))
            print(" contains NaN:", np.isnan(X).any(), "contains Inf:", np.isinf(X).any())
            # compute Jacobian eigenvalues at final state (1-D x)
            try:
                Jf = evaljacobianf(final, p, np.array([0.5,0.1]))
                eigs = np.linalg.eigvals(Jf)
                print(" Jacobian eigenvals (real parts):", np.round(eigs.real,6))
            except Exception as e:
                print(" Jacobian evaluation failed:", e)
        except Exception as e:
            print(" Could not inspect X:", e)
