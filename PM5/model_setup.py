"""
Helper functions to set up the 12-genotype bacterial model.

Reuses the parameter setup from PM4/antibiotic-simulation for consistency.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import mutation matrix creation from PM4
# Note: antibiotic-simulation has a hyphen, so we import directly from file
sys.path.insert(0, str(REPO_ROOT / "PM4" / "antibiotic-simulation"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "antibiotic_sweep",
    REPO_ROOT / "PM4" / "antibiotic-simulation" / "antibiotic-sweep.py"
)
antibiotic_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(antibiotic_sweep)

create_mutation_matrix = antibiotic_sweep.create_mutation_matrix
create_resistance_parameters = antibiotic_sweep.create_resistance_parameters


def setup_12_genotype_model(mutation_type='fitness_landscape', 
                            uR=0.5, uC=0.1,
                            resistance_range=(0.1, 5.0)):
    """
    Set up the 12-genotype bacterial model with standard parameters.
    
    Parameters
    ----------
    mutation_type : str
        Type of mutation matrix ('fitness_landscape' is default)
    uR : float
        Resource supply rate (default: 0.5)
    uC : float
        Antibiotic supply rate (default: 0.1)
    resistance_range : tuple
        (min_IC50, max_IC50) range (default: (0.1, 5.0))
    
    Returns
    -------
    p : dict
        Model parameters
    x0 : array, shape (14,)
        Initial state [n1, n2, ..., n12, R, C]
    eval_u : callable
        Input function u(t) = [uR, uC]
    """
    m = 12
    
    # Create parameters using PM4's function
    p = create_resistance_parameters(
        m=m,
        resistance_range=resistance_range,
        mutation_structure='gradient',
        mutation_type=mutation_type,
        base_mutation_rate=1e-8,
        population_size=1e7
    )
    
    # Initial state: uniform populations, reasonable R and C
    x0 = np.array([10.0] * m + [1.0, 0.2])
    
    # Input function (constant in time for now)
    def eval_u(t):
        return np.array([uR, uC])
    
    return p, x0, eval_u

