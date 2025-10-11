# tests/conftest.py
"""
pytest conftest: ensure project root is on sys.path and force non-interactive mpl backend.

Place this file in the tests/ directory. It runs early during pytest collection and:
 - inserts the project root (parent of tests/) into sys.path so tests can import local modules
   via `import tools.something` or `import linearize`.
 - sets matplotlib backend to 'Agg' to avoid GUI/interactive windows during tests (SimpleSolver).
 - optionally sets a reproducible numpy print/seed policy for stable outputs.
"""

import os
import sys

# 1) Make sure project root (parent of tests/) is on sys.path
THIS_DIR = os.path.dirname(__file__)           # path/to/project/tests
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    # insert at front so local modules shadow any same-name site packages
    sys.path.insert(0, PROJECT_ROOT)

# 2) Force matplotlib to use a non-interactive backend (avoid GUI windows during tests)
# Do this before any test imports matplotlib.pyplot or other plotting.
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    # If matplotlib is not installed, nothing to do
    pass

# 3) Small niceties to make outputs reproducible / less noisy in tests
try:
    import numpy as np
    np.set_printoptions(precision=6, suppress=True)
except Exception:
    pass

# (Optional) Provide a fixture for common model params if you want.
# Example:
# import pytest
# @pytest.fixture
# def basic_model_params():
#     m = 3
#     return {
#         'Q': np.eye(m),
#         'rmax': np.ones(m),
#         'K': 0.5 * np.ones(m),
#         'alpha': 0.1 * np.ones(m),
#         'd0': 0.2 * np.ones(m),
#         'IC50': np.ones(m),
#         'h': np.ones(m),
#         'kC': 0.05
#     }
