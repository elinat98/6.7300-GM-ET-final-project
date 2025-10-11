#!/usr/bin/env python3
"""
find_omega_limits.py  -- robust, GUI-suppressed version

Usage:
    python find_omega_limits.py
    python find_omega_limits.py --NumIter 200 --w_start 1e-2 --rel_tol 1e-6 --save results.json

Notes:
 - Place this file in your project root (same directory as tools/ or SimpleSolver.py).
 - The script tries multiple import strategies so it works whether or not tools/ is a package.
"""

# ----------------------------- GUI suppression -----------------------------
import os
import sys
import types
from pathlib import Path

# Force non-interactive matplotlib backend before any plotting imports
os.environ.setdefault("MPLBACKEND", "Agg")

# Provide dummy VisualizeState module (no GUI) so imports in SimpleSolver won't pop windows
if 'VisualizeState' not in sys.modules:
    mod_vs = types.ModuleType('VisualizeState')
    class DummyVisualizeState:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
        def show(self, *a, **k): return None
        def update(self, *a, **k): return None
        def close(self, *a, **k): return None
        def set_data(self, *a, **k): return None
    mod_vs.VisualizeState = DummyVisualizeState
    # also allow tools.VisualizeState import resolution
    sys.modules['VisualizeState'] = mod_vs
    sys.modules['tools.VisualizeState'] = mod_vs
# ---------------------------------------------------------------------------

# Ensure repo root is on sys.path (script directory)
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------------------------- robust imports -----------------------------
import importlib.util
import importlib
import math
import time
import json
import argparse
import numpy as np

def load_module_from_path(module_name, file_path):
    """Load a module from a .py file and insert it into sys.modules under module_name."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # remove partially loaded module on error
        sys.modules.pop(module_name, None)
        raise
    return module

# Try multiple ways to obtain SimpleSolver and evalf
SimpleSolver = None
evalf = None

# 1) direct imports
try:
    from SimpleSolver import SimpleSolver as _SS
    SimpleSolver = _SS
except Exception:
    pass

try:
    from evalf_bacterial import evalf as _ef
    evalf = _ef
except Exception:
    pass

# 2) package-style imports (tools.)
if SimpleSolver is None:
    try:
        from tools.SimpleSolver import SimpleSolver as _SS
        SimpleSolver = _SS
    except Exception:
        pass

if evalf is None:
    try:
        from tools.evalf_bacterial import evalf as _ef
        evalf = _ef
    except Exception:
        pass

# 3) fallback: load from files under tools/ or repo root
if SimpleSolver is None:
    # candidates for SimpleSolver file
    cand1 = REPO_ROOT / "tools" / "SimpleSolver.py"
    cand2 = REPO_ROOT / "SimpleSolver.py"
    found = None
    for c in (cand1, cand2):
        if c.exists():
            found = c
            break
    if found:
        mod = load_module_from_path("SimpleSolver_fallback", found)
        if hasattr(mod, "SimpleSolver"):
            SimpleSolver = getattr(mod, "SimpleSolver")

if evalf is None:
    cand1 = REPO_ROOT / "tools" / "evalf_bacterial.py"
    cand2 = REPO_ROOT / "evalf_bacterial.py"
    found = None
    for c in (cand1, cand2):
        if c.exists():
            found = c
            break
    if found:
        mod = load_module_from_path("evalf_bacterial_fallback", found)
        if hasattr(mod, "evalf"):
            evalf = getattr(mod, "evalf")

# Final sanity check
if SimpleSolver is None or evalf is None:
    raise ImportError("Failed to import SimpleSolver and/or evalf. "
                      "Ensure SimpleSolver.py and evalf_bacterial.py exist in project root or tools/.")

# ----------------------------- utility functions -----------------------------
def run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=200, w=1e-2):
    """
    Run solver; return (ok_flag, X, t, message).
    ok_flag False indicates exception, NaN/Inf, or explosion.
    """
    try:
        # try keyword call first
        try:
            res = solver_fn(evalf_fn, x0, p, u_func, NumIter=NumIter, w=w)
        except TypeError:
            res = solver_fn(evalf_fn, x0, p, u_func, NumIter, w)
    except Exception as e:
        return False, None, None, f"solver exception: {type(e).__name__}: {e}"

    if not (isinstance(res, tuple) and len(res) >= 2):
        return False, None, None, f"solver returned unexpected object of type {type(res)}"

    X, t = res[0], res[1]
    try:
        X = np.asarray(X, dtype=float)
    except Exception as e:
        return False, None, None, f"could not convert solver output X to array: {e}"

    if np.isnan(X).any() or np.isinf(X).any():
        return False, X, t, "NaN or Inf found in solver output"

    max_abs = float(np.max(np.abs(X)))
    if not np.isfinite(max_abs) or max_abs > 1e8:
        return False, X, t, f"solution exploded: max abs={max_abs:.3e}"

    return True, X, t, "ok"

def final_state_from_X(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return X.ravel()
    return X[:, -1].ravel()

def rel_error(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na = np.linalg.norm(a)
    if na == 0:
        return np.linalg.norm(a-b)
    return np.linalg.norm(a-b) / na

# ----------------------------- omega search routines -----------------------------
def find_omega_max(evalf_fn, solver_fn, x0, p, u_func, NumIter=200,
                   w0=1e-3, grow_factor=2.0, max_w_cap=1e8, tol_factor=1.01, max_exp_iters=60):
    # ensure at least one stable point
    w = float(w0)
    ok, X, t, msg = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w)
    if not ok:
        # try decreasing to find anchor
        w_try = w
        for _ in range(20):
            w_try /= 10.0
            ok2, X2, t2, msg2 = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w_try)
            if ok2:
                w = w_try
                ok, X, t, msg = ok2, X2, t2, msg2
                break
        if not ok:
            return None, {"error": "No stable omega found near w0."}

    w_stable = w
    w_curr = w_stable
    w_bad = None
    for _ in range(max_exp_iters):
        w_next = w_curr * grow_factor
        if w_next > max_w_cap:
            break
        ok_next, Xn, tn, msgn = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w_next)
        if ok_next:
            w_curr = w_next
            w_stable = w_curr
            continue
        else:
            w_bad = w_next
            break

    if w_bad is None:
        return w_stable, {"reason": "reached cap without failure", "w_stable": w_stable}

    # bisect (geometric) to refine boundary
    w_low = w_stable
    w_high = w_bad
    for _ in range(50):
        if w_high / w_low <= tol_factor:
            break
        w_mid = math.sqrt(w_low * w_high)
        ok_mid, Xm, tm, msgm = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w_mid)
        if ok_mid:
            w_low = w_mid
        else:
            w_high = w_mid
    omega_max = w_low
    return omega_max, {"w_bad": w_bad, "w_stable_initial": w_stable, "w_high": w_high, "w_low": w_low}

def find_omega_min(evalf_fn, solver_fn, x0, p, u_func, NumIter=200,
                   rel_tol=1e-6, w_start=1e-2, shrink_factor=2.0,
                   min_w_cap=1e-14, max_exp_iters=60):
    # find reference at small w_ref
    w_ref = min(1e-6, w_start/10.0)
    ref_ok, Xref, tref, msgref = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter*2, w=w_ref)
    if not ref_ok:
        # try increasing w_ref until stable
        w_try = w_ref
        found = False
        for _ in range(12):
            w_try *= 10.0
            ok_try, Xtry, ttry, msgtry = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w_try)
            if ok_try:
                ref_ok, Xref, tref, msgref = ok_try, Xtry, ttry, msgtry
                w_ref = w_try
                found = True
                break
        if not found:
            return None, {"error": "Could not compute reference solution for omega_min search."}

    x_ref_final = final_state_from_X(Xref)

    w_high = max(w_start, w_ref)
    ok_high, Xhigh, thigh, msghigh = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w_high)
    if not ok_high:
        # attempt to find some stable high
        w_temp = w_high
        found = False
        for _ in range(12):
            w_temp /= 2.0
            ok_temp, Xtemp, ttemp, msgtemp = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w_temp)
            if ok_temp:
                w_high = w_temp
                Xhigh = Xtemp
                ok_high = True
                found = True
                break
        if not found:
            return None, {"error": "Could not find stable starting w_high for omega_min search."}

    w_low = w_high
    w_bad = None
    for _ in range(max_exp_iters):
        w_next = w_low / shrink_factor
        if w_next < min_w_cap:
            break
        ok_next, Xn, tn, msgn = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w_next)
        if not ok_next:
            w_bad = w_next
            break
        x_final = final_state_from_X(Xn)
        err = rel_error(x_ref_final, x_final)
        if err > rel_tol:
            w_bad = w_next
            break
        else:
            w_low = w_next

    if w_bad is None:
        return w_low, {"reason": "no bad found; reached min cap", "w_low": w_low}

    # bisect geometrically between w_good and w_fail
    w_good = w_low
    w_fail = w_bad
    for _ in range(50):
        if w_fail / w_good <= 1.01:
            break
        w_mid = math.sqrt(w_good * w_fail)
        ok_mid, Xmid, tmid, msgmid = run_solver_safe(evalf_fn, solver_fn, x0, p, u_func, NumIter=NumIter, w=w_mid)
        if not ok_mid:
            w_fail = w_mid
            continue
        xmid = final_state_from_X(Xmid)
        err_mid = rel_error(x_ref_final, xmid)
        if err_mid <= rel_tol:
            w_good = w_mid
        else:
            w_fail = w_mid

    omega_min = w_good
    return omega_min, {"w_ref": w_ref, "w_good": w_good, "w_fail": w_fail, "rel_tol": rel_tol}

# ----------------------------- default model / main -----------------------------
def default_model_params():
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
    x_flat = np.array([10.0, 5.0, 2.0, 1.0, 0.2])
    x0_col = x_flat.reshape((-1, 1))
    u = lambda t: np.array([0.5, 0.1])
    return p, x0_col, u

def main(argv=None):
    parser = argparse.ArgumentParser(description="Find omega_min and omega_max for SimpleSolver on your model.")
    parser.add_argument("--NumIter", type=int, default=200, help="Number of iterations/time steps for SimpleSolver.")
    parser.add_argument("--w_start", type=float, default=1e-2, help="Starting w for searches.")
    parser.add_argument("--rel_tol", type=float, default=1e-6, help="Relative tolerance for omega_min.")
    parser.add_argument("--max_w_cap", type=float, default=1e8, help="Max omega cap.")
    parser.add_argument("--min_w_cap", type=float, default=1e-14, help="Min omega cap.")
    parser.add_argument("--save", type=str, default="omega_limits.json", help="Save results JSON (or 'none').")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    p, x0_col, u = default_model_params()
    print("Using default model and x0. Edit script to change. Repo root:", REPO_ROOT)
    print(f"NumIter={args.NumIter}, w_start={args.w_start}, rel_tol={args.rel_tol}")

    print("\nFinding omega_max (instability threshold)...")
    t0 = time.time()
    omega_max, info_max = find_omega_max(evalf, SimpleSolver, x0_col, p, u, NumIter=args.NumIter, w0=args.w_start, max_w_cap=args.max_w_cap)
    t1 = time.time()
    print(f"omega_max estimate = {omega_max}")
    print("info_max:", info_max)
    print(f"Elapsed {t1-t0:.3f}s")

    print("\nFinding omega_min (smallest w that matches reference within tol)...")
    t0 = time.time()
    omega_min, info_min = find_omega_min(evalf, SimpleSolver, x0_col, p, u, NumIter=args.NumIter, rel_tol=args.rel_tol, w_start=args.w_start, min_w_cap=args.min_w_cap)
    t1 = time.time()
    print(f"omega_min estimate = {omega_min}")
    print("info_min:", info_min)
    print(f"Elapsed {t1-t0:.3f}s")

    # recommended omega: between 10x above min and 0.1x below max
    recommended = None
    if omega_min is not None and omega_max is not None:
        rec1 = omega_min * 10.0
        rec2 = omega_max / 10.0
        if rec1 <= rec2:
            recommended = math.sqrt(rec1 * rec2)
        else:
            recommended = min(rec2, max(rec1, 1e-12))

    summary = {
        "omega_max": float(omega_max) if omega_max is not None else None,
        "omega_min": float(omega_min) if omega_min is not None else None,
        "recommended": float(recommended) if recommended is not None else None,
        "info_max": info_max,
        "info_min": info_min
    }

    print("\nSummary:")
    print("  omega_max =", summary["omega_max"])
    print("  omega_min =", summary["omega_min"])
    print("  recommended omega =", summary["recommended"])

    if args.save and args.save.lower() != "none":
        Path(args.save).write_text(json.dumps(summary, indent=2))
        print("Saved results to", args.save)

    return summary

if __name__ == "__main__":
    main()
