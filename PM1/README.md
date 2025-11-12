## PM1: Model evaluation and jacobian 
The model describes bacterial subpopulations competing for a shared resource **R** and subject to an antibiotic concentration **C**. Each subpopulation grows, consumes resources, and dies according to nonlinear Monod–Hill kinetics.

### Implementation 

### `evalf_bacterial.py`

Defines the core nonlinear model function `evalf(x, p, u)` which:

* Supports both 1-D and column vector inputs
* Handles safe Monod and Hill computations even at limits ( R\to0 ) or ( C\to0 )
* Returns ( f(x,p,u) ) with consistent shape (column or flat)

Validated for dimensional consistency and finite outputs across parameter ranges.

### `jacobian_tools.py`

Implements the **analytic Jacobian** ( J = \frac{\partial f}{\partial x} ) using explicit derivatives of all nonlinear terms (Monod and Hill).
Includes a testbench `jacobian_testbench()` comparing analytic vs. finite-difference Jacobians.


### `jacobian_condition_checks.py`

Runs full trajectory simulations using a **SimpleSolver** and checks conditioning of the Jacobian along the path:

* Computes condition number and minimum singular value vs. time
* Performs 2D scans over (R, C) to locate near-singular regions.

### `testjacobianplot.py`

Performs a **finite-difference dx-sweep** comparing the analytic Jacobian against an external FD reference:

* Generates CSV table (`jacobian_error_table_externalFD.csv`)
* Produces log–log plot (`jacobian_error_vs_dx_scalar_externalFD.png`)
* Automatically identifies optimal dx minimizing Frobenius error.

### `visualize_jacobian_sparsity.py`

Visualizes the **sparsity pattern** of the Jacobian matrix for sparsity analysis:

* Creates sparsity pattern plot (spy plot) showing non-zero elements
* Generates magnitude heatmap (log scale) of Jacobian values
* Annotates block structure (∂f/∂n, ∂f/∂R, ∂f/∂C, etc.)
* Prints detailed sparsity statistics (density, block-wise analysis, condition number)

**Usage:**
```bash
# Basic usage (m=3 subpopulations, default state)
python -m PM1.visualize_jacobian_sparsity

# Custom number of subpopulations
python -m PM1.visualize_jacobian_sparsity --m 12

# Save figure to file
python -m PM1.visualize_jacobian_sparsity --m 12 --save jacobian_sparsity.png

# Custom state and input vectors
python -m PM1.visualize_jacobian_sparsity --m 3 --x "10,5,2,1,0.2" --u "0.5,0.1"

# Statistics only (no plots)
python -m PM1.visualize_jacobian_sparsity --stats-only
```

### `analyze_jacobian_singularity.py`

Analyzes **where the Jacobian becomes singular** and explains condition number magnitudes:

* Scans (R, C) parameter space to find worst-conditioned regions
* Analyzes extreme cases (low resource, high antibiotic, zero populations)
* Provides intuitive explanations for why singularity occurs
* Creates visualizations of condition number and singular values across parameter space
* See `JACOBIAN_SINGULARITY_ANALYSIS.md` for detailed findings

**Usage:**
```bash
# Full analysis with visualizations
python -m PM1.analyze_jacobian_singularity --m 3

# Larger system
python -m PM1.analyze_jacobian_singularity --m 12 --n-points 100

# Save figure
python -m PM1.analyze_jacobian_singularity --save singularity_analysis.png
```

**Key Findings:**
- Condition numbers typically range from 10² to 10⁴ (well to moderately conditioned)
- Worst-conditioned at low resource (R ≈ 0.27, C = 0) with κ ≈ 7×10⁴
- Along trajectories: median κ ≈ 2,000, max κ ≈ 25,000
- System becomes singular when birth rates → 0 (low R or high C) or populations → 0

### `plot_evolution_over_time.py`

Plots the **evolution of R (resource), C (antibiotic), and all genotypes over time**:

* Runs a simulation using SimpleSolver
* Creates comprehensive visualizations showing:
  - All genotype populations over time (log scale)
  - Resource (R) and antibiotic (C) concentrations over time
  - Total biomass and diversity metrics (Shannon entropy, Simpson diversity)
  - Genotype fractions over time (optional)
* Useful for understanding system dynamics and convergence to steady state

**Usage:**
```bash
# Basic usage (m=3, 400 time steps)
python -m PM1.plot_evolution_over_time

# Custom number of subpopulations and time steps
python -m PM1.plot_evolution_over_time --m 12 --NumIter 500

# Custom initial state and inputs
python -m PM1.plot_evolution_over_time --m 3 --x0 "10,5,2,1,0.2" --u "0.5,0.1"

# Save figure
python -m PM1.plot_evolution_over_time --save evolution.png

# Skip fractions plot
python -m PM1.plot_evolution_over_time --no-fractions
```

### `pytests (in ../tests folder)`
Pytests to exhastively test the model function evalf exists at `test_evalf_extended.py` and will handle zero negative or extreme values , ie if R = 0 (births shut down, only death), IC50 = 0 (full inhibition under antibiotic pressure), a = 0 (ensures zero resource consumption) and confirms broadcast handling for parameter arrays and correct signs - ie non negative populations 

Pytest to test the jacobian conditions exist at `test_jacobian_condition_quick.py` which runs a short simulation with the SimpleSolver and checks the conditions of the jacobian at the final state. Confirms all singular values are finite and validates the jacobian is well posted throughout the trajectory 

### Quick start

```bash
# 1. Run trajectory + Jacobian conditioning
python -m PM1.jacobian_condition_checks.py

# 2. Run finite-difference Jacobian comparison
python -m PM1.testjacobianplot.py

# 3. Visualize Jacobian sparsity pattern
python -m PM1.visualize_jacobian_sparsity --m 12 --save jacobian_sparsity.png

# 4. Analyze Jacobian singularity and condition numbers
python -m PM1.analyze_jacobian_singularity --m 3 --save singularity_analysis.png

# 5. Plot evolution of R, C, and genotypes over time
python -m PM1.plot_evolution_over_time --m 3 --save evolution_over_time.png

# 6. Run pytest for evalf function 
pytest ../tests/test_evalf_extended.py 

#Run pytest for jacobian conditioning 
pytest ../tests/test_jacobian_condition_quick.py 
```



