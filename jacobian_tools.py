# jacobian_tools.py
import numpy as np

def evaljacobianf(x, p, u):
    """
    Analytic Jacobian J = df/dx for the bacterial evolution model.
    x: array length m+2 -> [n1..nm, R, C]
    p: dict with keys Q, rmax, K, alpha, d0, IC50, h, kC
    u: inputs (not used in Jacobian)
    Returns: J shape (m+2, m+2)
    """
    x = np.asarray(x, dtype=float)
    m = x.size - 2
    if m <= 0:
        raise ValueError("x must have length >= 3 (m>=1)")
    n = x[:m]
    R = x[m]
    C = x[m+1]

    # Force shapes to vectors with length m
    Q = np.asarray(p['Q'], dtype=float)
    rmax = np.asarray(p['rmax'], dtype=float).reshape(m,)
    K = np.asarray(p['K'], dtype=float).reshape(m,)
    alpha = np.asarray(p['alpha'], dtype=float).reshape(m,)
    d0 = np.asarray(p['d0'], dtype=float).reshape(m,)
    IC50 = np.asarray(p['IC50'], dtype=float).reshape(m,)
    h = np.asarray(p['h'], dtype=float).reshape(m,)
    kC = float(p['kC'])

    # Monod term and derivative 
    denom = K + R
    monod = np.where(denom != 0.0, R / denom, 0.0)
    dmonod_dR = np.where(denom != 0.0, K / (denom**2), 0.0)  # derivative of R/(K+R)

    # Hill term and derivative wrt C (safe handling of IC50==0) 
    ratio = np.zeros(m)
    nonzero_ic50 = IC50 != 0.0
    if np.any(nonzero_ic50):
        ratio[nonzero_ic50] = (C / IC50[nonzero_ic50]) ** h[nonzero_ic50] if C != 0 else 0.0
    if np.any(~nonzero_ic50):
        # if IC50 == 0 and C>0, treat ratio -> inf => hill -> 0
        ratio[~nonzero_ic50] = np.inf if C > 0 else 0.0
    hill = 1.0 / (1.0 + ratio)

    # dhill/dC safe computation (correct limit at C==0)
    dhill_dC = np.zeros(m, dtype=float)
    # mask for entries with IC50 != 0 (we handle IC50==0 separately)
    nonzero_ic50 = IC50 != 0.0
    mask = nonzero_ic50.copy()

    if np.any(mask):
        # compute s = (C/IC50)^h safely
        s = np.zeros(m, dtype=float)
        if C != 0.0:
            s[mask] = (C / IC50[mask]) ** h[mask]
        else:
            # at C==0, s = 0 for h>0
            s[mask] = 0.0

        # compute ds/dC with correct C->0 limits:
        ds_dC = np.zeros(m, dtype=float)
        # entries where h == 1 --> ds/dC = 1/IC50 even at C==0
        idx_h1 = mask & (np.isclose(h, 1.0))
        if np.any(idx_h1):
            ds_dC[idx_h1] = 1.0 / IC50[idx_h1]
        # entries where h > 1 --> ds/dC = 0 at C==0 (already zero)
        # entries where C != 0 and h arbitrary --> use analytic formula
        idx_Cnonzero = mask & (C != 0.0)
        if np.any(idx_Cnonzero):
            ds_dC[idx_Cnonzero] = h[idx_Cnonzero] * (C / IC50[idx_Cnonzero]) ** (h[idx_Cnonzero] - 1) / IC50[idx_Cnonzero]

        # final dhill/dC
        dhill_dC = - ds_dC / (1.0 + s)**2

    # birth rates and their derivatives
    b = rmax * monod * hill
    db_dR = rmax * dmonod_dR * hill
    db_dC = rmax * monod * dhill_dC

    # Allocate jacobian matrix (full size = (m+2)x(m+2))
    N = m + 2
    J = np.zeros((N, N), dtype=float)

    # Top-left block: ∂f_i / ∂n_k = b_k * Q[k,i] - d_i * delta(i,k) - > death term is subtracted on the diagonal 
    for i in range(m):
        for k in range(m):
            J[i, k] = b[k] * Q[k, i]
        J[i, i] -= d0[i]

    # Top-right: ∂f_i/∂R and ∂f_i/∂C
    # ∂f_i/∂R = sum_j n_j * db_j/dR * Q[j,i] -> chain rule + similarity for C 
    for i in range(m):
        J[i, m] = np.sum((n * db_dR) * Q[:, i])
        J[i, m+1] = np.sum((n * db_dC) * Q[:, i])

    # R row - > resource index (row index = m) 
    J[m, :m] = - alpha * b                       # ∂R_dot/∂n_k = -alpha_k * b_k
    J[m, m] = -1.0 - np.sum(alpha * n * db_dR)  # ∂R_dot/∂R
    J[m, m+1] = - np.sum(alpha * n * db_dC)     # ∂R_dot/∂C

    # C row -> antibiotic index (row index = m+ 1) 
    J[m+1, m+1] = -1.0 - kC

    return J


def jacobian_testbench(f_eval, x, p, u, dx_factors=None):
    """
    Compare analytic Jacobian to finite-difference across dx scaling factors.
    Returns list of (dx_factor, frobenius_norm_difference).
    dx_factors: array-like multipliers applied to the 'scaled' base dx
    """
    if dx_factors is None:
        dx_factors = np.logspace(-1, -8, 8)

    J_analytic = evaljacobianf(x, p, u)
    results = []
    eps = np.finfo(float).eps

    # base_dx per component (scaled)
    base_dx = 2.0 * np.sqrt(eps) * np.maximum(1.0, np.abs(x))
    for fac in dx_factors:
        dx_vec = base_dx * fac
        # central difference using per-component dx
        N = x.size
        y0 = f_eval(x, p, u)
        M = y0.size
        J_fd = np.zeros((M, N))
        for k in range(N):
            ek = np.zeros(N); ek[k] = 1.0
            dxk = dx_vec[k]
            fk_plus = f_eval(x + dxk * ek, p, u)
            fk_minus = f_eval(x - dxk * ek, p, u)
            J_fd[:, k] = (fk_plus - fk_minus) / (2.0 * dxk)
        diff_norm = np.linalg.norm(J_fd - J_analytic, ord='fro')
        results.append((fac, diff_norm))
    return results
