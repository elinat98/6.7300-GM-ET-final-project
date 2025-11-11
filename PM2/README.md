## PM2: Conditioning, Linearization and Multiple RHS 
This milestone extends PM1 by analyzing numerical conditioning, linearizing the nonlinear bacterial model, and implementing efficient multiple-RHS steady-state solvers. 

PM2 builds on the nonlinear bacterial model developed in PM1 by adding:
- Numerical conditioning analysis to identify stable integration step sizes.
- Finite-difference diagnostics to understand round-off and truncation error.
- A linearization routine returning \(A\) and \(B\) matrices at \((x_0, u_0)\).
- Efficient multiple-right-hand-side (RHS) solvers for steady-state predictions.

### Implementation 

### `find_omega_limits.py`
Searches for the **safe time-step range (ω)** for the SimpleSolver.
- Iteratively increases ω until numerical instability is detected (divergence or NaN).
- Then decreases ω until accuracy deviation exceeds a tolerance.
- Outputs:
  - `omega_min`, `omega_max`, and `recommended` safe ω range.
  - Writes results to `omega_limits.json`.

Run:
```bash
python -m PM2.find_omega_limits --NumIter 200 --rel_tol 1e-6
```
### `jacobian_fd_sweep.py**`
Sweeps the finite-difference step size `dx` and compares the analytic Jacobian vs finite-difference (FD) approximation:
- Computes Frobenius norm error ‖J_fd – J_analytic‖ for each dx.
- Plots error vs dx on a log–log scale and identifies optimal dx.
- Demonstrates the trade-off between truncation error (large dx) and round-off (small dx).
- Outputs:   
- `jacobian_error_vs_dx_scalar_externalFD.png` — error vs dx plot  
- `jacobian_error_table_externalFD.csv` — numeric table

Run:
```bash
python -m PM2.jacobian_fd_sweep
```


### `linearize.py`
Implements a general-purpose linearization routine that computes:
\[
A = \frac{\partial f}{\partial x}\Big|_{x_0, u_0}, \quad
B = [K_0 \;|\; J_u]
\]
where \( K_0 = f(x_0,u_0) - A x_0 - J_u u_0 \).
- Uses analytic Jacobian (`evaljacobianf`) when available.
- Falls back to finite-difference estimation otherwise.
- Supports central or forward differencing on inputs \(u\).
- Produces consistent \(A\) and \(B\) matrices for linearized system analysis.

Example:
```python
from linearize import linearize_f
from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf

x0 = [10,5,2,1,0.2]
u0 = [0.5,0.1]
p = default_params()

A, B = linearize_f(evalf, evaljacobianf, x0, p, u0, du=1e-6)
```

#### `multiple_rhs.py`
Provides vectorized solvers for systems with multiple simultaneous right-hand-sides:
- `solve_multiple_rhs(A, RHS)` → Reuses one LU factorization for all RHS columns.
- `steady_state_solutions(A, B, U)` → Computes steady-state \(x_{ss}\) for multiple input vectors \(u\).
- `condition_number(A)` → Computes matrix condition number κ₂(A).

Purpose: accelerate steady-state evaluations for multiple parameter inputs or control cases.

Example:
```python
from multiple_rhs import steady_state_solutions, condition_number
x_ss = steady_state_solutions(A, B, U_array)
print("Condition number:", condition_number(A))
```

### `pytests (in ../tests folder)`

The following regression and verification tests are included for PM2.  
They can be executed together with:

```bash
pytest -v -k "test_linearize or test_multiple_rhs or test_jacobian_condition_quick or test_jacobian_external_fd"
```

#### `test_linearize.py`
Validates the **linearization function** by checking that `linearize_f()` correctly computes \(A\) and \(B\) matrices for a known system.  
Confirms the numerical derivatives in \(A\) and \(B\) match expected analytic results, ensuring that the local linear model \( \dot{x} = A x + B u \) is consistent with the nonlinear dynamics.

#### `test_multiple_rhs.py`
Verifies the **multiple-RHS steady-state solver**.  
Tests that:
- `solve_multiple_rhs()` reuses one LU factorization efficiently across several RHS vectors.  
- `steady_state_solutions()` produces correct steady states \( A x = -B[1;u] \).  
- `condition_number(A)` remains within reasonable limits, confirming well-posed solutions.

#### `test_jacobian_condition_quick.py`
Performs a **Jacobian conditioning check** at the final solver state.  
Runs a short integration, computes \( J_f(x^\*) \), and ensures:
- All singular values are finite.  
- Condition number \( \kappa(J_f) \) < 1e12.  
- The Jacobian remains well-conditioned across the stable ω range.

#### `test_jacobian_external_fd.py`
Compares the **analytic Jacobian** to the **finite-difference Jacobian** over a range of perturbation sizes (`dx`).  
Plots and reports:
- Frobenius norm error ‖J_fd − J_analytic‖.  
- Identifies the optimal FD step size balancing truncation and round-off errors.  
This test directly supports PM2 task (C) *Conditioning: Analysis & Improvement*.

To run all: ```bash 
pytest -v -k "test_linearize or test_multiple_rhs or test_jacobian_condition_quick or test_jacobian_external_fd 
```

---
### Quick start
```bash
python -m PM2.find_omega_limits --NumIter 200 --rel_tol 1e-6
python -m PM2.jacobian_fd_sweep
pytest -v -k "test_linearize or test_multiple_rhs or test_jacobian_condition_quick or test_jacobian_external_fd"
```


### Notes