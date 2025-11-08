# tools/jacobian_fd_sweep.py
import numpy as np
import matplotlib.pyplot as plt
from jacobian_tools import evaljacobianf
from evalf_bacterial import evalf

m = 3
p = {
    'Q': np.eye(m),
    'rmax': np.array([1.2, 0.8, 0.5]),
    'K': np.zeros(m),
    'alpha': np.zeros(m),
    'd0': np.array([0.3,0.2,0.1]),
    'IC50': np.array([1.0,1.0,1.0]),
    'h': np.ones(m),
    'kC': 0.05
}
x = np.array([5.0, 3.0, 1.0, 1.0, 0.0])
J_analytic = evaljacobianf(x, p, np.array([0.0, 0.0]))

# base dx (scaled) and wide factors (10^5 down to 10^-20 if needed)
eps = np.finfo(float).eps
base_dx = 2.0 * np.sqrt(eps) * np.maximum(1.0, np.abs(x))
dx_factors = np.logspace(5, -20, num=120)

dxs = []
errs = []
for fac in dx_factors:
    dx_vec = base_dx * fac
    # central difference
    N = x.size
    M = J_analytic.shape[0]
    J_fd = np.zeros((M,N))
    for k in range(N):
        ek = np.zeros(N); ek[k]=1.0
        dxk = dx_vec[k]
        fk_plus = np.asarray(evalf(x + dxk*ek, p, np.array([0.0,0.0]))).ravel()
        fk_minus = np.asarray(evalf(x - dxk*ek, p, np.array([0.0,0.0]))).ravel()
        J_fd[:,k] = (fk_plus - fk_minus) / (2.0*dxk)
    err = np.linalg.norm(J_fd - J_analytic, ord='fro')
    dxs.append(np.exp(np.mean(np.log(np.abs(dx_vec)))))
    errs.append(err)

dxs = np.array(dxs); errs = np.array(errs)
imin = np.argmin(errs)
print("best dx (geom mean) = ", dxs[imin], "err=", errs[imin])

plt.loglog(dxs, errs, '-o')
plt.xlabel('absolute dx (geom mean)')
plt.ylabel('Frobenius error')
plt.grid(True, which='both')
plt.axvline(dxs[imin], color='r', linestyle='--', label='best dx')
plt.legend()
plt.show()
