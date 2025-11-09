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


### `pytests (in ../tests folder)`
Pytests to exhastively test the model function evalf exists at `test_evalf_extended.py` and will handle zero negative or extreme values , ie if R = 0 (births shut down, only death), IC50 = 0 (full inhibition under antibiotic pressure), a = 0 (ensures zero resource consumption) and confirms broadcast handling for parameter arrays and correct signs - ie non negative populations 

Pytest to test the jacobian conditions exist at `test_jacobian_condition_quick.py` which runs a short simulation with the SimpleSolver and checks the conditions of the jacobian at the final state. Confirms all singular values are finite and validates the jacobian is well posted throughout the trajectory 

### Quick start

```bash
# 1. Run trajectory + Jacobian conditioning
python -m PM1.jacobian_condition_checks.py

# 2. Run finite-difference Jacobian comparison
python -m PM1. testjacobianplot.py

#3 Run pytest for evalf function 
pytest ../tests/test_evalf_extended.py 

#Run pytest for jacobian conditioning 
pytest ../tests/test_jacobian_condition_quick.py 
```



