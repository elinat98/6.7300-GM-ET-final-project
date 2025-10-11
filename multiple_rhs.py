# multiple_rhs.py
"""
Utilities for solving linear systems with multiple right-hand sides,
and for computing steady states of the linearized model:

Given linearization:
    dx/dt = A x + B [1; u]

The steady-state solves 0 = A x + B [1; u] -> A x = - B [1; u]

This module provides functions to compute the steady-state x for many
different u inputs efficiently (multiple RHS).

Functions:
- solve_multiple_rhs(A, RHS) -> X  # solves A X = RHS for possibly many RHS columns
- steady_state_solutions(A, B, U) -> X or X_matrix
- condition_number(A) -> float
"""

from typing import Tuple
import numpy as np

def condition_number(A: np.ndarray) -> float:
    """
    Compute the 2-norm condition number of A.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array")
    return np.linalg.cond(A)


def solve_multiple_rhs(A: np.ndarray, RHS: np.ndarray) -> np.ndarray:
    """
    Solve A X = RHS for X. Accepts RHS as shape (N,) or (N,k).
    Returns X with the same "column" layout: if RHS was (N,) returns (N,),
    if RHS was (N,k) returns (N,k).

    Uses numpy.linalg.solve which accepts a 2-D RHS and solves all columns
    in one call (efficient: underlying LAPACK factorization reused).
    """
    A = np.asarray(A, dtype=float)
    RHS = np.asarray(RHS, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (N,N)")
    N = A.shape[0]

    if RHS.ndim == 1:
        if RHS.size != N:
            raise ValueError("RHS length does not match A.shape[0]")
        # np.linalg.solve accepts 1-D RHS but returns 1-D solution
        X = np.linalg.solve(A, RHS)
        return X
    elif RHS.ndim == 2:
        if RHS.shape[0] != N:
            raise ValueError("RHS shape[0] must equal A.shape[0]")
        # Solve for all columns at once
        X = np.linalg.solve(A, RHS)
        return X
    else:
        raise ValueError("RHS must be 1-D or 2-D array")


def steady_state_solutions(A: np.ndarray, B: np.ndarray, U: np.ndarray):
    """
    Compute steady-state solutions x solving 0 = A x + B [1; u] for one or many u.

    Parameters
    ----------
    A : (N,N) array
        Linearization matrix.
    B : (N, 1 + m_u) array
        B columns arranged as [K0 | Ju] where K0 is constant offset and Ju are
        input Jacobian columns (N x m_u).
    U : array-like
        Can be:
          - shape (m_u,) representing a single input vector u
          - shape (m_u, K) representing K different u vectors (each column a u)
          - shape (K, m_u) representing K different u vectors (each row a u)

    Returns
    -------
    x_ss : (N,) if single u passed
           (N, K) if multiple u passed (columns correspond to each u case)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    U = np.asarray(U, dtype=float)

    # dimension checks
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (N,N)")
    N = A.shape[0]

    if B.ndim != 2 or B.shape[0] != N:
        raise ValueError("B must have shape (N, 1 + m_u)")

    m_u = B.shape[1] - 1
    if m_u < 0:
        raise ValueError("B must have at least one column (K0)")

    # Normalize U to shape (m_u, K)
    if U.ndim == 1:
        if U.size != m_u:
            raise ValueError(f"Single u must have length m_u={m_u}")
        U_mat = U.reshape((m_u, 1))
    elif U.ndim == 2:
        # Accept either (m_u, K) or (K, m_u)
        if U.shape[0] == m_u:
            U_mat = U.copy()
        elif U.shape[1] == m_u:
            U_mat = U.T.copy()
        else:
            raise ValueError(f"U must have shape (m_u, K) or (K, m_u); m_u={m_u}")
    else:
        raise ValueError("U must be 1-D or 2-D array")

    Kcases = U_mat.shape[1]

    # Compose the stacked [1; u] matrix of shape (1 + m_u, Kcases)
    ones_row = np.ones((1, Kcases), dtype=float)
    aug = np.vstack([ones_row, U_mat])   # shape (1+m_u, Kcases)

    # Compute RHS = - B @ aug  -> shape (N, Kcases)
    RHS = - B.dot(aug)

    # Solve A X = RHS  (multiple RHS columns solved in one call)
    # Before solving, check conditioning
    condA = np.linalg.cond(A)
    if not np.isfinite(condA):
        raise np.linalg.LinAlgError("Condition number of A is infinite or NaN (A singular?)")
    # If condition is large, we still attempt to solve but warn
    if condA > 1e12:
        # small non-intrusive warning via print; user can catch logs elsewhere
        print(f"Warning: large condition number for A: cond(A) = {condA:.3e}")

    X = solve_multiple_rhs(A, RHS)   # shape (N, Kcases)

    if Kcases == 1:
        return X.ravel()   # return 1-D vector for single u
    return X               # return (N, Kcases) matrix
