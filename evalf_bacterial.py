# evalf_bacterial.py
import numpy as np
import warnings

def evalf(x, p, u):
    """
    Evaluate f(x, p, u) for the bacterial evolution model:
        x = [n_1, ..., n_m, R, C]  (length m+2)
        p is a dict with required keys:
            'Q'    : (m,m) mutation kernel (rows: parent j -> cols: offspring i)
            'rmax' : (m,) array-like r_{j,max}
            'K'    : (m,) array-like Monod half-saturation constants K_j
            'alpha': (m,) array-like resource consumption per newborn alpha_j
            'd0'   : (m,) array-like baseline death rates d_i
            'IC50' : (m,) array-like IC50_j
            'h'    : (m,) array-like Hill exponent h_j
            'kC'   : scalar antibiotic decay rate
        u is array-like length 2: [uR, uC]

    Returns:
        f : ndarray length m+2 = [n_dot (m), R_dot, C_dot]

    Behavior notes:
    - Q rows will be validated/normalized automatically:
        * If a row sums to zero -> set that row to identity (no-mutation).
        * If rows do not sum to 1, rows are normalized and a warning is emitted.
    - Handles sparse Q (scipy.sparse) by converting to dense for this function.
    - Safely handles K + R == 0 and IC50 == 0 edge cases.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1D array-like of shape (m+2,)")

    m = x.size - 2
    if m < 1:
        raise ValueError("x must contain at least one genotype plus R and C (length >= 3).")


    n = x[:m]          
    R = float(x[m])    
    C = float(x[m+1])  

    # Required parameter keys
    required = ['Q', 'rmax', 'K', 'alpha', 'd0', 'IC50', 'h', 'kC']
    for k in required:
        if k not in p:
            raise KeyError(f"Parameter dictionary p must contain key '{k}'")

    # -- handle Q (dense or sparse) and normalize/validate rows --
    Q_raw = p['Q']
    # detect scipy.sparse if available
    try:
        from scipy.sparse import issparse
    except Exception:
        def issparse(x): return False

    if issparse(Q_raw):
        try:
            Q = Q_raw.toarray()
        except Exception as e:
            raise ValueError("Provided sparse Q could not be converted to dense array: " + str(e))
    else:
        Q = np.asarray(Q_raw, dtype=float)

    if Q.shape != (m, m):
        raise ValueError(f"Q must have shape ({m},{m}); got {Q.shape}")

    # Row-sum handling
    row_sums = Q.sum(axis=1)


    # If any row sums are NaN or infinite, error out
    if np.any(~np.isfinite(row_sums)):
        raise ValueError("Q contains non-finite elements or produced non-finite row sums.")

    # Fix rows that sum to zero: set to identity (no mutation)
    zero_rows = np.isclose(row_sums, 0.0)
    if np.any(zero_rows):
        warnings.warn("Some rows of Q sum to zero. Setting those rows to identity (no mutation).", RuntimeWarning)
        Q = Q.copy()
        for j in np.where(zero_rows)[0]:
            Q[j, :] = 0.0
            Q[j, j] = 1.0
        row_sums = Q.sum(axis=1)


    # Normalize rows if they do not sum to 1 (within tolerance)
    if not np.allclose(row_sums, 1.0, rtol=1e-8, atol=1e-12):
        warnings.warn("Q rows are not row-stochastic (do not sum to 1). Normalizing rows automatically.", RuntimeWarning)
        Q = Q / row_sums[:, np.newaxis]


    # -- unpack & validate other params --
    rmax = np.asarray(p['rmax'], dtype=float).reshape(-1)
    K = np.asarray(p['K'], dtype=float).reshape(-1)
    alpha = np.asarray(p['alpha'], dtype=float).reshape(-1)
    d0 = np.asarray(p['d0'], dtype=float).reshape(-1)
    IC50 = np.asarray(p['IC50'], dtype=float).reshape(-1)
    h = np.asarray(p['h'], dtype=float).reshape(-1)
    kC = float(p['kC'])

    for name, arr in (('rmax', rmax), ('K', K), ('alpha', alpha), ('d0', d0), ('IC50', IC50), ('h', h)):
        if arr.size not in (m,):
            # allow (m,1) or (1,m) by reshape earlier; make strict
            raise ValueError(f"Parameter '{name}' must be length {m}")

    # -- compute birth rates b_j(R,C) safely (vectorized) --
    # Monod term: R / (K_j + R)
    denom = K + R  # length m

    # avoid division by zero
    monod = np.zeros_like(denom, dtype=float)
    nonzero_denom = denom != 0.0
    monod[nonzero_denom] = R / denom[nonzero_denom]
    # when denom == 0 -> treat monod as 0 (no growth)

    # Hill inhibition: 1 / (1 + (C/IC50)^h)
    # handle IC50==0: if IC50==0 and C>0 -> ratio -> inf -> hill -> 0
    ratio = np.zeros(m, dtype=float)
    nonzero_ic50 = IC50 != 0.0
    if np.any(nonzero_ic50):
        # safe power; if C==0 -> (0/IC50)^h == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio[nonzero_ic50] = (C / IC50[nonzero_ic50]) ** h[nonzero_ic50] if C != 0 else 0.0
    if np.any(~nonzero_ic50):
        ratio[~nonzero_ic50] = np.inf if C > 0 else 0.0
    hill = 1.0 / (1.0 + ratio)

    b = rmax * monod * hill   # length m

    # death rate
    d = d0

    # genotype dynamics
    # births by parent j: n_j * b_j
    births_by_parent = n * b   # length m
    # births into each offspring i: sum_j births_by_parent[j] * Q[j,i]
    # (matrix multiply: (1xm) @ (mxm) -> (m,) vector)
    births_into_i = births_by_parent @ Q  # length m

    n_dot = births_into_i - d * n

    # resource dynamics
    uR = float(u[0])
    consumption = np.sum(alpha * n * b)   # sum_j alpha_j * n_j * b_j
    R_dot = uR - R - consumption

    # antibiotic dynamics
    uC = float(u[1])
    C_dot = uC - C - kC * C

    return np.concatenate([n_dot, np.array([R_dot, C_dot])])
