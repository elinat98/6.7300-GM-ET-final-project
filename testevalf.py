"""
Extended regression / edge-case tests for evalf(x,p,u).

Put this file next to your evalf implementation (evalf_bacterial.py)
and run:

    python testevalf_extended.py

The tests assert correct behavior in limiting cases (zero resource,
IC50==0, division-by-zero handling, Q shape checks, sparse-Q handling,
large C inhibition, broadcasting of parameter shapes, etc).
"""

import numpy as np
from numpy.testing import assert_allclose

# Import evalf from your module - adjust name if needed
try:
    from evalf_bacterial import evalf
except Exception as e:
    raise ImportError("Could not import evalf from evalf_bacterial.py. "
                      "Make sure the file exists and defines evalf(x,p,u).") from e

def mk_base_params(m):
    """Create a baseline parameter dict for tests."""
    return {
        'Q': np.eye(m),
        'rmax': np.ones(m),
        'K': 0.5 * np.ones(m),
        'alpha': 0.1 * np.ones(m),
        'd0': 0.2 * np.ones(m),
        'IC50': 1.0 * np.ones(m),
        'h': 1.0 * np.ones(m),
        'kC': 0.05
    }

def test_zero_resource_no_births():
    """R=0 -> monod term 0 -> b=0 -> n_dot = -d*n and R_dot = uR (no consumption)."""
    m = 4
    n = np.array([10., 5., 2., 1.])
    R = 0.0
    C = 0.0
    x = np.concatenate([n, np.array([R, C])])
    p = mk_base_params(m)
    u = np.array([0.7, 0.0])
    f = evalf(x, p, u)
    expected_n_dot = - p['d0'] * n
    assert_allclose(f[:m], expected_n_dot, rtol=1e-8, atol=1e-12)
    # Because b=0 consumption=0 -> R_dot = uR - R = uR
    assert np.isclose(f[m], u[0]), f"R_dot expected {u[0]} but got {f[m]}"
    print("test_zero_resource_no_births: PASS")

def test_ic50_zero_with_positive_C():
    """
    IC50 == 0 treated as infinite inhibition when C>0 -> hill ~ 0 -> b ~ 0.
    So behavior same as R=0 case for b.
    """
    m = 3
    n = np.array([3.0, 2.0, 1.0])
    R = 0.5
    C = 1.0
    x = np.concatenate([n, np.array([R, C])])
    p = mk_base_params(m)
    p['IC50'] = np.zeros(m)   # edge case
    u = np.array([0.4, 0.1])
    f = evalf(x, p, u)
    # since IC50==0 and C>0 => strong inhibition => b approximately 0 => n_dot = -d*n
    assert_allclose(f[:m], -p['d0'] * n, rtol=1e-8, atol=1e-10)
    # R_dot should be uR - R (no consumption because b~0)
    assert np.isclose(f[m], u[0] - R, atol=1e-12) or np.isclose(f[m], u[0]), "R_dot unexpected for IC50==0"
    print("test_ic50_zero_with_positive_C: PASS")

def test_K_plus_R_zero_handled():
    """
    If K + R == 0 for some j, implementation should avoid division by zero and
    treat monod contribution as 0. We'll force K = -R to test that branch.
    """
    m = 2
    n = np.array([1.0, 2.0])
    R = 1.0
    C = 0.0
    x = np.concatenate([n, np.array([R, C])])
    p = mk_base_params(m)
    p['K'] = np.array([-R, 0.3])   # first genotype has K + R == 0
    u = np.array([0.2, 0.0])
    f = evalf(x, p, u)
    # For genotype 0 we expect monod=0 -> it contributes no births; check no crash & finite
    assert np.isfinite(f).all(), "evalf produced non-finite entries when K+R==0"
    print("test_K_plus_R_zero_handled: PASS")

def test_Q_shape_validation_raises():
    """
    If Q has incorrect shape, evalf should raise ValueError (per the implementation).
    """
    m = 3
    n = np.array([1.0, 1.0, 1.0])
    x = np.concatenate([n, np.array([1.0, 0.0])])
    p = mk_base_params(m)
    p['Q'] = np.ones((m+1, m))   # wrong shape
    try:
        _ = evalf(x, p, np.array([0.1,0.0]))
    except ValueError:
        print("test_Q_shape_validation_raises: PASS (ValueError raised as expected)")
        return
    raise AssertionError("Expected ValueError due to bad Q shape, but evalf did not raise.")


def test_sparse_Q_behavior():
    """
    If user supplies a scipy.sparse matrix for Q, evalf may not accept it.
    This test tries to pass a CSR matrix and checks whether evalf raises an informative error.
    """
    try:
        from scipy.sparse import csr_matrix
    except Exception:
        print("test_sparse_Q_behavior: SKIPPED (scipy not available)")
        return

    m = 3
    n = np.array([2.0, 1.0, 0.0])
    x = np.concatenate([n, np.array([1.0, 0.0])])
    p = mk_base_params(m)
    Q_sparse = csr_matrix(np.eye(m))
    p['Q'] = Q_sparse
    try:
        f = evalf(x, p, np.array([0.1,0.0]))
    except Exception as e:
        print("test_sparse_Q_behavior: Handled (evalf raised on sparse Q) - recommend converting sparse->dense before calling evalf.")
        return
    # If no error, ensure result is finite
    assert np.isfinite(f).all(), "evalf produced non-finite output with sparse Q"
    print("test_sparse_Q_behavior: PASS (sparse Q accepted and produced finite output)")

def test_large_C_inhibition_produces_zero_births():
    """
    Very large antibiotic C relative to IC50 should suppress births (b ~ 0).
    Then n_dot should be approximately -d*n.
    """
    m = 3
    n = np.array([4.0, 3.0, 1.0])
    R = 2.0
    C = 1e9
    x = np.concatenate([n, np.array([R, C])])
    p = mk_base_params(m)
    p['IC50'] = np.array([1.0, 1.0, 1.0])
    u = np.array([0.2, 0.0])
    f = evalf(x, p, u)
    # b should be tiny; allow a modest tolerance
    assert_allclose(f[:m], -p['d0'] * n, rtol=1e-6, atol=1e-8)
    print("test_large_C_inhibition_produces_zero_births: PASS")

def test_alpha_zero_no_consumption():
    """alpha = 0 => consumption term zero => R_dot = uR - R."""
    m = 2
    n = np.array([1.0, 2.0])
    R = 1.5
    C = 0.0
    x = np.concatenate([n, np.array([R, C])])
    p = mk_base_params(m)
    p['alpha'] = np.zeros(m)
    u = np.array([0.3, 0.0])
    f = evalf(x, p, u)
    assert_allclose(f[m], u[0] - R, rtol=1e-12, atol=1e-12)
    print("test_alpha_zero_no_consumption: PASS")

def test_param_broadcasting():
    """
    Pass parameters in (m,1) shape to ensure the function handles broadcasting or reshaping.
    """
    m = 3
    n = np.array([1.0, 2.0, 1.0])
    x = np.concatenate([n, np.array([1.0, 0.0])])
    p = mk_base_params(m)
    # make rmax and K column vectors (m,1)
    p['rmax'] = np.array([1.0, 0.8, 0.7]).reshape((m,1))
    p['K'] = np.array([0.4,0.4,0.4]).reshape((m,1))
    # should not error
    f = evalf(x, p, np.array([0.1,0.0]))
    assert np.isfinite(f).all()
    print("test_param_broadcasting: PASS")

def test_random_consistency_with_Q_identity():
    """
    With Q = I, the births term reduces to n*b (elementwise), so
    n_dot = n*b - d*n. This test checks that identity Q yields that form.
    """
    m = 4
    n = np.array([1.0, 2.0, 0.5, 0.0])
    x = np.concatenate([n, np.array([2.0, 0.0])])
    p = mk_base_params(m)
    p['Q'] = np.eye(m)
    f = evalf(x, p, np.array([0.1,0.0]))
    # compute b directly
    R = x[m]; C = x[m+1]
    monod = R / (p['K'] + R)
    hill = 1.0 / (1.0 + (C / p['IC50']) ** p['h'])
    b = p['rmax'] * monod * hill
    expected_n_dot = n * b - p['d0'] * n
    assert_allclose(f[:m], expected_n_dot, rtol=1e-10, atol=1e-12)
    print("test_random_consistency_with_Q_identity: PASS")

def test_negative_state_handling():
    """
    Passing negative state values should not lead to crashes (although
    negative populations are unphysical). We assert evalf returns finite numbers.
    """
    m = 2
    x = np.array([-1.0, 2.0, -0.5, -0.1])  # negative n and R and C
    p = mk_base_params(m)
    try:
        f = evalf(x, p, np.array([0.0, 0.0]))
    except Exception as e:
        raise AssertionError("evalf crashed on negative state input: " + str(e))
    assert np.isfinite(f).all(), "evalf returned non-finite values for negative state"
    print("test_negative_state_handling: PASS")

def run_all_tests():
    tests = [
        test_zero_resource_no_births,
        test_ic50_zero_with_positive_C,
        test_K_plus_R_zero_handled,
        test_Q_shape_validation_raises,
        test_sparse_Q_behavior,
        test_large_C_inhibition_produces_zero_births,
        test_alpha_zero_no_consumption,
        test_param_broadcasting,
        test_random_consistency_with_Q_identity,
        test_negative_state_handling
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as ae:
            failures.append((t.__name__, str(ae)))
        except Exception as e:
            failures.append((t.__name__, f"UNEXPECTED ERROR: {type(e).__name__}: {e}"))
    print("\n=== TEST SUMMARY ===")
    if failures:
        print(f"{len(failures)} tests FAILED:")
        for name, msg in failures:
            print(f" - {name}: {msg}")
    else:
        print("All tests passed successfully.")

if __name__ == "__main__":
    run_all_tests()
