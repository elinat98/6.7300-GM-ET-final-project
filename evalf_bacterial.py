# evalf_bacterial.py  (updated evalf)
import numpy as np
import warnings

def evalf(x, p, u):
    """
    Evaluate f(x,p,u) for the bacterial evolution model.

    Compatible input shapes for x:
      - 1-D array shape (N,)  -> returns 1-D array shape (N,)
      - column array shape (N,1) -> returns column array shape (N,1)
    where N = m + 2

    p : dict with keys 'Q','rmax','K','alpha','d0','IC50','h','kC'
    u : array-like length 2 -> [uR, uC]
    """
    # --- normalize input shapes ---
    x_in = np.asarray(x, dtype=float)
    was_column = (x_in.ndim == 2 and x_in.shape[1] == 1)
    if was_column:
        x_flat = x_in.ravel()         # shape (N,)
    else:
        x_flat = x_in.ravel()

    # Basic checks
    N = x_flat.size
    if N < 3:
        raise ValueError("x must have length >= 3 (m>=1)")
    m = N - 2

    # Unpack state (1-D)
    n = x_flat[:m].astype(float)
    R = float(x_flat[m])
    C = float(x_flat[m+1])

    # -- read params and coerce shapes --
    # allow scalars or arrays; enforce length m vectors
    Q = np.asarray(p['Q'], dtype=float)
    if Q.shape != (m, m):
        raise ValueError(f"Q must be shape ({m},{m}); got {Q.shape}")

    def _as1d(name):
        a = np.asarray(p[name], dtype=float)
        return a.reshape(m,) if a.size == m else np.full(m, float(a)) if a.size == 1 else a.reshape(m,)

    rmax = _as1d('rmax')
    K    = _as1d('K')
    alpha= _as1d('alpha')
    d0   = _as1d('d0')
    IC50 = _as1d('IC50')
    h    = _as1d('h')
    kC   = float(p.get('kC', 0.0))

    # --- safe Monod and derivatives ---
    denom = K + R
    with np.errstate(divide='ignore', invalid='ignore'):
        monod = np.where(denom != 0.0, R / denom, 0.0)
        dmonod_dR = np.where(denom != 0.0, K / (denom**2), 0.0)

    # --- safe Hill (C dependence) and derivative ---
    ratio = np.zeros(m, dtype=float)
    nonzero_ic50 = (IC50 != 0.0)
    if np.any(nonzero_ic50):
        if C != 0.0:
            ratio[nonzero_ic50] = (C / IC50[nonzero_ic50]) ** h[nonzero_ic50]
        else:
            ratio[nonzero_ic50] = 0.0
    if np.any(~nonzero_ic50):
        ratio[~nonzero_ic50] = np.inf if C > 0 else 0.0
    hill = 1.0 / (1.0 + ratio)

    # derivative dhill/dC
    dhill_dC = np.zeros(m, dtype=float)
    mask = nonzero_ic50 & (IC50 != 0.0)
    if np.any(mask):
        s = np.zeros(m, dtype=float)
        if C != 0.0:
            s[mask] = (C / IC50[mask]) ** h[mask]
        else:
            s[mask] = 0.0
        ds_dC = np.zeros(m, dtype=float)
        valid = mask & (h != 0)
        if np.any(valid):
            if C != 0.0:
                ds_dC[valid] = h[valid] * (C / IC50[valid]) ** (h[valid] - 1) / IC50[valid]
            else:
                ds_dC[valid] = 0.0
        dhill_dC = - ds_dC / (1.0 + s)**2

    # --- birth rates and derivatives ---
    b = rmax * monod * hill
    db_dR = rmax * dmonod_dR * hill
    db_dC = rmax * monod * dhill_dC

    # --- compute f components ---
    # births_by_parent = n * b  (length m)
    births_by_parent = n * b
    # births into each offspring i: sum_j births_by_parent[j] * Q[j,i]
    births_into = births_by_parent @ Q    # length m

    n_dot = births_into - d0 * n          # vector length m
    uR = float(u[0]) if hasattr(u, "__len__") else float(u)
    uC = float(u[1]) if (hasattr(u, "__len__") and len(u) > 1) else 0.0

    consumption = np.sum(alpha * n * b)
    R_dot = uR - R - consumption
    C_dot = uC - C - kC * C

    f_flat = np.concatenate([n_dot, np.array([R_dot, C_dot])], axis=0)

    # return in matching shape: column if input was column, else 1-D
    if was_column:
        return f_flat.reshape((N,1))
    else:
        return f_flat
