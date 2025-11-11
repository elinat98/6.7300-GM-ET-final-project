# tests/test_multiple_rhs.py
import numpy as np
import numpy.testing as npt
from PM2.multiple_rhs import steady_state_solutions, solve_multiple_rhs, condition_number

def make_invertible_matrix(N, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.randn(N, N) * 0.5
    # add diagonal dominance to ensure invertible and well-conditioned-ish
    A = A + np.eye(N) * (1.5 + 0.1 * N)
    return A

def test_solve_multiple_rhs_basic():
    N = 5
    A = make_invertible_matrix(N, seed=1)
    # create 3 RHS columns
    rng = np.random.RandomState(2)
    RHS = rng.randn(N, 3)
    X = solve_multiple_rhs(A, RHS)
    # check equal to naive loop
    X_loop = np.column_stack([np.linalg.solve(A, RHS[:, i]) for i in range(RHS.shape[1])])
    npt.assert_allclose(X, X_loop, rtol=1e-12, atol=1e-12)

def test_steady_state_many_inputs():
    N = 4
    m_u = 2
    A = make_invertible_matrix(N, seed=5)
    rng = np.random.RandomState(7)
    Ju = rng.randn(N, m_u) * 0.3
    K0 = rng.randn(N) * 0.1
    B = np.hstack([K0.reshape(N,1), Ju])

    # build 4 random inputs (as rows)
    U_rows = rng.randn(4, m_u) * 0.2

    # compute steady states using the function
    X_mat = steady_state_solutions(A, B, U_rows)  # returns (N, K)
    assert X_mat.shape == (N, U_rows.shape[0]) or X_mat.shape == (N, U_rows.shape[0]).__class__  # sanity

    # compute naive solution per-case and compare
    X_loop = np.zeros_like(X_mat)
    for i in range(U_rows.shape[0]):
        u = U_rows[i, :]
        rhs = - B.dot(np.hstack([1.0, u]))
        xvec = np.linalg.solve(A, rhs)
        X_loop[:, i] = xvec

    npt.assert_allclose(X_mat, X_loop, rtol=1e-12, atol=1e-12)

def test_condition_number_warning_and_return():
    N = 3
    # make nearly singular matrix by small singular value
    U, _ = np.linalg.qr(np.random.RandomState(1).randn(N, N))
    V, _ = np.linalg.qr(np.random.RandomState(2).randn(N, N))
    s = np.array([1.0, 1e-6, 1e-9])  # small singular values -> large cond
    A = (U * s) @ V.T
    # simple B, U
    B = np.zeros((N, 1 + 1))
    U0 = np.array([0.0])
    # condition number function should run
    condA = condition_number(A)
    assert condA > 1e6
    # steady_state_solutions will still attempt to solve; for extremely singular may raise
    try:
        x = steady_state_solutions(A, B, U0)
        # if returned, verify shape
        assert x.shape[0] == N
    except np.linalg.LinAlgError:
        # acceptable if solver raises due to singularity
        pass
