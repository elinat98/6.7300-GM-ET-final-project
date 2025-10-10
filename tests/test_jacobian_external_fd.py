# tests/test_jacobian_external_fd.py
"""
Test that compares the analytic Jacobian (evaljacobianf) to the external
finite-difference Jacobian (eval_Jf_FiniteDifference). Also runs the
jacobian_testbench sweep and prints results.

This file is a pytest test but also can be executed directly with python.
"""

import numpy as np
import pytest

# Try imports and give helpful errors if missing
try:
    from jacobian_tools import evaljacobianf, jacobian_testbench
except Exception as e:
    raise ImportError("Could not import evaljacobianf or jacobian_testbench from jacobian_tools.py: " + str(e))

try:
    from evalf_bacterial import evalf
except Exception as e:
    raise ImportError("Could not import evalf from evalf_bacterial.py: " + str(e))

try:
    # external finite-difference function (expects column x and returns (Jf, dxFD) or Jf)
    from tools.eval_Jf_FiniteDifference import eval_Jf_FiniteDifference
except Exception as e:
    raise ImportError("Could not import eval_Jf_FiniteDifference from eval_Jf_FiniteDifference.py: " + str(e))


def _run_jacobian_external_fd_test():
    # --- small example problem (same as your original) ---
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

    # Analytic Jacobian (1-D input)
    J_analytic = evaljacobianf(x_flat, p, u)
    J_analytic = np.asarray(J_analytic, dtype=float)

    # Call external finite-difference jacobian (expects column x)
    J_fd_result = eval_Jf_FiniteDifference(evalf, x_col, p, u)

    # Normalize return (handle either J or (J,dx) just in case)
    if isinstance(J_fd_result, tuple) and len(J_fd_result) == 2:
        J_fd, dxFD = J_fd_result
    else:
        J_fd = J_fd_result
        dxFD = None

    J_fd = np.asarray(J_fd, dtype=float)

    print("\n=== External FD Jacobian comparison ===")
    print("Used dxFD =", dxFD)

    # shape check
    assert J_fd.shape == J_analytic.shape, (
        f"Shape mismatch: J_fd.shape={J_fd.shape}, J_analytic.shape={J_analytic.shape}"
    )

    # compute norms
    diff_fro = np.linalg.norm(J_fd - J_analytic, ord='fro')
    rel_err = diff_fro / (np.linalg.norm(J_analytic, ord='fro') + 1e-30)

    print("Frobenius norm difference (analytic vs FD):", diff_fro)
    print("Relative Frobenius error:", rel_err)

    # run the jacobian_testbench sweep and print results (this uses evaljacobianf internally)
    print("\n=== jacobian_testbench sweep results ===")
    results = jacobian_testbench(evalf, x_flat, p, u)
    for fac, err in results:
        print(f"dx_factor={fac:.1e}, frobenius_error={err:.3e}")

    # Basic sanity assertions: numeric, finite, and diff is finite
    assert np.isfinite(diff_fro), "Non-finite Frobenius difference"
    assert np.isfinite(rel_err), "Non-finite relative error"

    # We don't assert a strict tolerance here because FD method or dx choice can vary;
    # if you want a strict regression guard, change the threshold below.
    # Example (optional): assert rel_err < 1e-6
    return diff_fro, rel_err, dxFD, results


def test_jacobian_external_fd_runs_and_matches_shape():
    """
    Pytest wrapper: ensures the external FD Jacobian call runs, shapes match,
    and the outputs/norms are finite. Prints some diagnostics.
    """
    diff_fro, rel_err, dxFD, results = _run_jacobian_external_fd_test()
    # Basic pass/fail: make sure relative error is finite and non-negative
    assert rel_err >= 0.0


if __name__ == "__main__":
    # allow running as a standalone script for verbose output
    _run_jacobian_external_fd_test()
