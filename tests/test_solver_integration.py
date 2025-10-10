import numpy as np
from evalf_bacterial import evalf
from tools.SimpleSolver import SimpleSolver

def test_solver_runs_and_returns_reasonable_state():
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
    x0 = np.array([10.0, 5.0, 2.0, 1.0, 0.2]).reshape((-1, 1))

    def u(t):
        return np.array([0.5, 0.1])

    X, t = SimpleSolver(evalf, x0, p, u, NumIter=50, w=0.01)

    # basic sanity checks
    assert X.shape[0] == 5
    assert len(t) == X.shape[1]
    assert not np.isnan(X).any()
    final = X[:3, -1].ravel()
    # populations should be non-negative (allow tiny negative numerical noise)
    assert np.all(final >= -1e-8)