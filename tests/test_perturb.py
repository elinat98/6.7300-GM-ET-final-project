# tests/perturb_test.py
import numpy as np
from numpy.linalg import eig
from jacobian_tools import evaljacobianf
from tools.SimpleSolver import SimpleSolver
from evalf_bacterial import evalf

def test_perturbation_grows_along_unstable_mode():
    """
    Quick smoke test: run a short simulation, compute Jacobian at final state,
    perturb along the most unstable eigenvector and ensure a small perturbation
    either grows (if eigenvalue positive) or decays appropriately.
    This test only asserts the code runs end-to-end without errors.
    """
    # model params
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

    # initial condition
    x0 = np.array([10.0, 5.0, 2.0, 1.0, 0.2]).reshape((-1,1))

    def u(t):
        return np.array([0.5, 0.1])

    # run solver a bit to get a final state
    X, t = SimpleSolver(evalf, x0, p, u, NumIter=60, w=0.01)
    x_final = X[:, -1].ravel()

    # Jacobian and eigen-decomposition
    J = evaljacobianf(x_final, p, u(t[-1]))
    eigvals, eigvecs = eig(J)
    # pick most positive real part index
    idx = np.argmax(np.real(eigvals))
    # normalized real eigenvector
    v = np.real(eigvecs[:, idx])
    v = v / np.linalg.norm(v)

    # small perturbation
    eps = 1e-4
    x_pert = x_final + eps * v
    Xp, tp = SimpleSolver(evalf, x_pert.reshape((-1,1)), p, u, NumIter=80, w=0.01)

    # just assert solver returned with same shape and no NaNs
    assert Xp.shape[0] == X.shape[0]
    assert not np.isnan(Xp).any()
