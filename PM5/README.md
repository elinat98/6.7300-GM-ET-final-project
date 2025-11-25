## PM5: Explicit vs Implicit ODE Time Integrators

This milestone compares explicit (Forward Euler) and implicit ODE time integrators for the 12-genotype bacterial population dynamics model.

### Implementation

#### `forward_euler.py`
Implements the explicit Forward Euler method:
```
x[n+1] = x[n] + ∆t * f(x[n], p, u(t[n]))
```

Features:
- Fixed time step ∆t throughout simulation
- Handles both 1D and column vector states
- Adjusts final step to exactly reach `t_stop`

#### `reference_solution.py`
Computes a reference solution by progressive refinement:
- Starts with `dt_initial` and reduces by factor of 2 each iteration
- Convergence criterion: `||x(tstop)_∆ti - x(tstop)_∆ti-1||∞ < εref`
- Stops when converged OR time exceeds 5 minutes (default)
- Returns reference solution, `dt_ref`, and convergence history

#### `model_setup.py`
Helper function to set up the 12-genotype model:
- Uses PM4's parameter creation functions for consistency
- Returns parameters `p`, initial state `x0`, and input function `eval_u`

#### `compare_integrators.py`
Main script that:
1. Sets up 12-genotype model
2. Computes reference solution (with convergence checking)
3. Runs Forward Euler for different ∆t values
4. Compares results to reference (ready for implicit methods)

### Usage

#### Basic Usage
```bash
# Run full comparison (reference + Forward Euler sweep)
python -m PM5.compare_integrators

# With custom time range
python -m PM5.compare_integrators --t-stop 5.0

# With faster reference computation (for testing)
python -m PM5.compare_integrators --max-time 60 --convergence-tol 1e-5
```

#### Options
- `--t-start T_START`: Starting time (default: 0.0)
- `--t-stop T_STOP`: Stopping time (default: 10.0)
- `--dt-initial DT_INITIAL`: Initial dt for reference (default: 0.1)
- `--convergence-tol TOL`: Convergence tolerance εref (default: 1e-6)
- `--max-time SECONDS`: Max computation time for reference (default: 300 = 5 min)
- `--skip-reference`: Skip reference computation (faster testing)
- `--dt-values DT1,DT2,...`: Custom dt values to test

#### Example Output
```
======================================================================
ODE INTEGRATOR COMPARISON: 12-GENOTYPE BACTERIAL MODEL
======================================================================

[1/3] Setting up 12-genotype model...
  Model: 12 genotypes + resource + antibiotic = 14 states

[2/3] Computing reference solution...
  Iteration 1: dt = 0.1
  Iteration 2: dt = 0.05
    Error: ||x_∆ti - x_∆ti-1||∞ = 8.66e-03
  ...
  ✓ CONVERGED after N iterations
    Final dt: 1.25e-04
    Final error: 5.13e-07 < 1.00e-06

[3/3] Running Forward Euler for different dt values...
  [1/5] dt = 0.1, Error = 4.23e-02
  [2/5] dt = 0.05, Error = 1.70e-02
  ...
```

### Current Status

✅ **Completed:**
- Forward Euler integrator with fixed ∆t
- Reference solution computation with convergence checking
- Time limit enforcement (5 minutes default)
- Comparison framework for different ∆t values
- 12-genotype model setup

🔄 **Ready for Next Steps:**
- Implicit integrators (Backward Euler, Crank-Nicolson, etc.)
- Adaptive time-stepping
- Error analysis and visualization
- Performance comparisons (computation time vs accuracy)

### Design Decisions

1. **Convergence Criterion**: Uses infinity norm `||·||∞` as specified
2. **Time Step Reduction**: Factor of 2 (standard halving) for efficiency
3. **Time Limit**: 5 minutes maximum to prevent excessive computation
4. **Model Setup**: Reuses PM4's parameter functions for consistency

### Files

- `forward_euler.py` - Forward Euler implementation
- `reference_solution.py` - Reference solution computation
- `model_setup.py` - 12-genotype model helper
- `compare_integrators.py` - Main comparison script
- `README.md` - This file

---

*Ready for further instructions on implicit integrators and extended comparisons.*

