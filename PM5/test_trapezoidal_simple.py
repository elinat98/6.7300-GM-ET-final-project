#!/usr/bin/env python3
"""Simple test of Trapezoidal method."""

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from PM5.model_setup import setup_12_genotype_model
from PM5.stability_analysis import trapezoidal_fallback

# Set up model
p, x0, eval_u = setup_12_genotype_model()

print(f"Testing Trapezoidal with dt=0.1, t=[0, 1.0]")
print(f"Initial state: {x0[:5]}...")

try:
    X, t, stats = trapezoidal_fallback(
        evalf, evaljacobianf, x0, p, eval_u,
        0.0, 1.0, 0.1,
        init_method='feuler', verbose=True
    )
    if X is not None:
        print(f"Success! Final state: {X[:, -1][:5]}...")
        print(f"Steps: {len(t)-1}")
    else:
        print("Failed: X is None")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

