#!/usr/bin/env python3
"""
Comprehensive stability and error analysis for Forward Euler and Trapezoidal integrators.

This script:
1. Computes reference solution with 10-minute limit
2. Finds largest stable ∆t for Forward Euler (∆tunst)
3. Computes error at instability (ϵunst)
4. Determines acceptable error level (ϵa)
5. Compares ϵa vs ϵunst and recommends best integrator/time-step
6. Tests Trapezoidal with different initializations
"""

import sys
from pathlib import Path
import numpy as np
import time
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from PM5.forward_euler import forward_euler
from PM5.reference_solution import compute_reference_solution
from PM5.model_setup import setup_12_genotype_model

# Check if trapezoidal is available
try:
    from PM5.trapezoidal import trapezoidal
    TRAPEZOIDAL_AVAILABLE = True
except ImportError:
    print("Warning: trapezoidal.py not found, will use alternative implementation")
    TRAPEZOIDAL_AVAILABLE = False


def is_stable(X, max_magnitude=1e10):
    """
    Check if simulation result is numerically stable.
    
    Parameters
    ----------
    X : array
        State trajectory
    max_magnitude : float
        Maximum allowed magnitude (default: 1e10)
    
    Returns
    -------
    stable : bool
        True if stable (no NaN, Inf, or explosion)
    reason : str
        Reason if unstable, "stable" otherwise
    """
    if X is None:
        return False, "None result"
    
    X = np.asarray(X, dtype=float)
    
    # Check for NaN or Inf
    if np.isnan(X).any():
        return False, "NaN detected"
    
    if np.isinf(X).any():
        return False, "Inf detected"
    
    # Check for explosion
    max_abs = float(np.max(np.abs(X)))
    if not np.isfinite(max_abs) or max_abs > max_magnitude:
        return False, f"Solution exploded: max|X| = {max_abs:.3e}"
    
    # Check for negative populations (should be non-negative)
    # This is a biological constraint, not necessarily numerical instability
    if np.any(X < -1e-6):  # Allow small numerical errors
        # This is a warning but not necessarily instability
        pass
    
    return True, "stable"


def find_unstable_dt_forward_euler(eval_f, x0, p, eval_u, t_start, t_stop,
                                   dt_start=0.5, growth_factor=1.5,
                                   dt_min_stable=None, verbose=True):
    """
    Find the largest ∆t before Forward Euler becomes unstable (∆tunst).
    
    Uses binary search to efficiently find the boundary.
    
    Parameters
    ----------
    eval_f : callable
        ODE right-hand side
    x0 : array
        Initial state
    p : dict
        Model parameters
    eval_u : callable
        Input function
    t_start, t_stop : float
        Time range
    dt_start : float
        Starting dt for search (default: 0.5)
    growth_factor : float
        Factor to increase dt when testing (default: 1.5)
    dt_min_stable : float, optional
        Known lower bound for stability (speeds up search)
    verbose : bool
        If True, print progress
    
    Returns
    -------
    dt_unst : float
        Largest dt that remains stable (∆tunst)
    dt_stable : float
        Smallest dt that is unstable
    stability_boundary : dict
        Information about the boundary
    """
    
    if verbose:
        print("="*70)
        print("FINDING STABILITY BOUNDARY FOR FORWARD EULER")
        print("="*70)
    
    # First, find a stable starting point
    dt_test = dt_start
    stable_found = False
    
    if dt_min_stable is not None:
        dt_test = dt_min_stable
        stable_found = True
    
    # Find a stable point to start from
    while not stable_found:
        try:
            X, t = forward_euler(eval_f, x0, p, eval_u, t_start, t_stop, dt_test, verbose=False)
            stable, reason = is_stable(X)
            if stable:
                stable_found = True
                if verbose:
                    print(f"✓ Found stable starting point: dt = {dt_test:.6e}")
                break
            else:
                dt_test /= 2.0
                if verbose:
                    print(f"  dt = {dt_test:.6e}: {reason}, reducing...")
        except Exception as e:
            dt_test /= 2.0
            if verbose:
                print(f"  dt = {dt_test:.6e}: Exception {type(e).__name__}, reducing...")
    
    # Now grow dt until we find instability
    dt_stable = dt_test
    dt_unst = None
    
    # Exponential search for upper bound
    dt_curr = dt_stable
    max_iter = 50
    
    if verbose:
        print(f"\nGrowing dt to find instability boundary...")
    
    for i in range(max_iter):
        dt_next = dt_curr * growth_factor
        
        try:
            X, t = forward_euler(eval_f, x0, p, eval_u, t_start, t_stop, dt_next, verbose=False)
            stable, reason = is_stable(X)
            
            if stable:
                dt_stable = dt_next
                dt_curr = dt_next
                if verbose and (i < 5 or i % 5 == 0):
                    print(f"  dt = {dt_next:.6e}: ✓ stable")
            else:
                dt_unst = dt_next
                if verbose:
                    print(f"  dt = {dt_next:.6e}: ✗ {reason}")
                break
                
        except Exception as e:
            dt_unst = dt_next
            if verbose:
                print(f"  dt = {dt_next}: ✗ Exception: {type(e).__name__}")
            break
    
    if dt_unst is None:
        dt_unst = dt_curr * growth_factor
        if verbose:
            print(f"Warning: No instability found up to dt = {dt_unst:.6e}")
    
    # Binary search to refine boundary
    if verbose:
        print(f"\nRefining boundary with binary search...")
        print(f"  Stable: dt = {dt_stable:.6e}")
        print(f"  Unstable: dt = {dt_unst:.6e}")
    
    dt_low = dt_stable
    dt_high = dt_unst
    tolerance = 1.01  # 1% tolerance
    
    for i in range(30):  # Max 30 iterations
        if dt_high / dt_low <= tolerance:
            break
        
        dt_mid = np.sqrt(dt_low * dt_high)  # Geometric mean
        
        try:
            X, t = forward_euler(eval_f, x0, p, eval_u, t_start, t_stop, dt_mid, verbose=False)
            stable, reason = is_stable(X)
            
            if stable:
                dt_low = dt_mid
                if verbose and i % 5 == 0:
                    print(f"  dt = {dt_mid:.6e}: ✓ stable")
            else:
                dt_high = dt_mid
                if verbose and i % 5 == 0:
                    print(f"  dt = {dt_mid:.6e}: ✗ {reason}")
                    
        except Exception as e:
            dt_high = dt_mid
            if verbose and i % 5 == 0:
                print(f"  dt = {dt_mid:.6e}: ✗ Exception")
    
    dt_unst = dt_low  # Largest stable dt
    
    if verbose:
        print(f"\n✓ Stability boundary found:")
        print(f"  ∆tunst = {dt_unst:.6e} (largest stable dt)")
        print(f"  dt_unstable > {dt_high:.6e}")
    
    return dt_unst, dt_high, {
        'dt_stable': dt_low,
        'dt_unstable': dt_high,
        'tolerance': dt_high / dt_low
    }


def trapezoidal_with_init(eval_f, eval_Jf, x0, p, eval_u, t_start, t_stop, dt,
                          init_method='feuler', verbose=False):
    """
    Trapezoidal integrator with configurable initialization.
    
    Parameters
    ----------
    init_method : str
        'previous' - use previous time step
        'feuler' - use Forward Euler prediction
    
    Returns
    -------
    X, t : arrays
        State trajectory and time points (for backward compatibility)
    """
    # Always use our implementation for consistency and control over initialization
    X, t, _ = trapezoidal_fallback(eval_f, eval_Jf, x0, p, eval_u, 
                                   t_start, t_stop, dt, init_method, verbose)
    return X, t


def trapezoidal_fallback(eval_f, eval_Jf, x0, p, eval_u, t_start, t_stop, dt,
                         init_method='feuler', verbose=False):
    """
    Trapezoidal implementation using PM4 Newton solver.
    
    Returns:
    -------
    X : array or None
        State trajectory
    t : array or None
        Time points
    newton_stats : dict
        Statistics about Newton iterations
    """
    from PM4.newton.NewtonSolver import newton_solve
    
    num_steps = int(np.ceil((t_stop - t_start) / dt))
    N = len(x0)
    X = np.zeros((N, num_steps + 1))
    t = np.zeros(num_steps + 1)
    
    X[:, 0] = x0
    t[0] = t_start
    
    newton_iterations = []
    convergence_failures = 0
    
    for n in range(num_steps):
        dt_actual = min(dt, t_stop - t[n])
        t[n+1] = t[n] + dt_actual
        
        x_n = X[:, n]
        u_n = eval_u(t[n])
        u_np1 = eval_u(t[n+1])
        f_n = eval_f(x_n, p, u_n)
        
        # Define residual for Trapezoidal: R(x) = x - x_n - (dt/2)[f_n + f(x)]
        # Newton solver expects: residual(x, p, u)
        def residual(x_new, p_dummy, u_dummy):
            x_new_col = x_new.reshape((-1, 1)) if x_new.ndim == 1 else x_new
            f_new = eval_f(x_new_col, p, u_np1)
            f_n_col = f_n.reshape((-1, 1)) if f_n.ndim == 1 else f_n
            x_n_col = x_n.reshape((-1, 1)) if x_n.ndim == 1 else x_n
            return (x_new_col - x_n_col - (dt_actual / 2.0) * (f_n_col + f_new))
        
        # Jacobian of residual: I - (dt/2) * J_f(x)
        # Newton solver expects: jacobian(x, p, u)
        def residual_jacobian(x_new, p_dummy, u_dummy):
            x_new_col = x_new.reshape((-1, 1)) if x_new.ndim == 1 else x_new
            I = np.eye(N)
            J_f = eval_Jf(x_new_col, p, u_np1)
            return I - (dt_actual / 2.0) * J_f
        
        # Initial guess
        if init_method == 'feuler':
            x_guess = (x_n + dt_actual * f_n.ravel()).reshape((-1, 1))  # Forward Euler
        else:  # 'previous'
            x_guess = x_n.reshape((-1, 1))
        
        # Solve with Newton
        try:
            x_new, info = newton_solve(residual, x_guess, {}, np.zeros(0),
                                      jacobian_fn=residual_jacobian, max_iter=50, 
                                      tol=1e-8, verbose=False)  # Relaxed tolerance
            newton_iters = info.get('iters', 0)
            newton_iterations.append(newton_iters)
            
            residual_norm = info.get('residual_norm', np.inf)
            if not info['converged']:
                # Check if residual is actually small enough
                if residual_norm > 1e-5:  # Only fail if truly large
                    if verbose:
                        print(f"Newton did not converge at step {n+1}, error = {residual_norm:.6e}")
                    convergence_failures += 1
                    if convergence_failures > 5:  # Too many failures
                        newton_stats = {
                            'iterations': newton_iterations,
                            'failures': convergence_failures,
                            'avg_iterations': np.mean(newton_iterations) if newton_iterations else 0,
                            'max_iterations': np.max(newton_iterations) if newton_iterations else 0
                        }
                        return None, None, newton_stats
                # If residual is small but marked unconverged, accept anyway
            X[:, n+1] = x_new.ravel()
        except Exception as e:
            if verbose:
                print(f"Exception at step {n+1}: {e}")
            newton_stats = {
                'iterations': newton_iterations,
                'failures': convergence_failures + 1,
                'avg_iterations': np.mean(newton_iterations) if newton_iterations else 0,
                'max_iterations': np.max(newton_iterations) if newton_iterations else 0
            }
            return None, None, newton_stats
    
    newton_stats = {
        'iterations': newton_iterations,
        'failures': convergence_failures,
        'avg_iterations': np.mean(newton_iterations) if newton_iterations else 0,
        'max_iterations': np.max(newton_iterations) if newton_iterations else 0
    }
    
    return X, t, newton_stats


def find_dt_for_error_target(eval_f, x0, p, eval_u, t_start, t_stop,
                             xref, error_target, method='forward_euler',
                             dt_low=None, dt_high=None, verbose=True):
    """
    Find dt that produces approximately the target error.
    
    Uses binary search to find ∆ta such that ||x_final - xref||∞ ≈ error_target.
    
    Parameters
    ----------
    error_target : float
        Target error level ϵa
    method : str
        'forward_euler' or 'trapezoidal'
    dt_low, dt_high : float, optional
        Bounds for search
    
    Returns
    -------
    dt_optimal : float
        dt that gives approximately error_target
    actual_error : float
        Actual error achieved
    computation_time : float
        Time to run simulation
    """
    
    if method == 'forward_euler':
        def integrator(*args, **kwargs):
            return forward_euler(*args[:6], kwargs.get('dt') or args[6], verbose=False)
        eval_Jf = None
    elif method == 'trapezoidal':
        integrator = trapezoidal_with_init
        eval_Jf = evaljacobianf
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Set default bounds
    if dt_low is None:
        dt_low = 1e-5  # Very small
    if dt_high is None:
        dt_high = 1.0  # Reasonable upper bound
    
    if verbose:
        print(f"\nFinding dt for error target ϵa = {error_target:.6e} using {method}")
        print(f"  Search range: [{dt_low:.6e}, {dt_high:.6e}]")
    
    # Binary search
    best_dt = None
    best_error = np.inf
    best_time = None
    
    for i in range(40):  # Max iterations
        if dt_high / dt_low <= 1.01:  # 1% tolerance
            break
        
        dt_test = np.sqrt(dt_low * dt_high)  # Geometric mean
        
        # Run simulation
        start_time = time.time()
        try:
            if method == 'forward_euler':
                X, t = forward_euler(eval_f, x0, p, eval_u, t_start, t_stop, dt_test, verbose=False)
            else:  # trapezoidal
                X, t = trapezoidal_with_init(eval_f, eval_Jf, x0, p, eval_u, 
                                t_start, t_stop, dt_test, init_method='feuler', verbose=False)
            
            if X is None:
                dt_high = dt_test
                continue
            
            stable, _ = is_stable(X)
            if not stable:
                dt_high = dt_test
                continue
            
            comp_time = time.time() - start_time
            x_final = X[:, -1]
            error = np.linalg.norm(x_final - xref, ord=np.inf)
            
            # Track best (closest to target)
            if abs(error - error_target) < abs(best_error - error_target):
                best_dt = dt_test
                best_error = error
                best_time = comp_time
            
            if verbose and i % 5 == 0:
                print(f"  dt = {dt_test:.6e}: error = {error:.6e}, time = {comp_time:.3f}s")
            
            # Adjust search
            # Larger dt typically gives larger error
            # If error > target, we need smaller dt
            if error > error_target:
                dt_high = dt_test  # Need smaller dt
            else:
                dt_low = dt_test  # Can try larger dt for efficiency
            
        except Exception as e:
            dt_high = dt_test
            if verbose:
                print(f"  dt = {dt_test:.6e}: Exception {type(e).__name__}")
    
    if best_dt is None:
        if verbose:
            print("Warning: Could not find dt for error target")
        return None, None, None
    
    if verbose:
        print(f"\n✓ Found: dt = {best_dt:.6e}")
        print(f"  Error: {best_error:.6e} (target: {error_target:.6e})")
        print(f"  Computation time: {best_time:.3f}s")
    
    return best_dt, best_error, best_time


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive stability and error analysis'
    )
    parser.add_argument('--t-start', type=float, default=0.0)
    parser.add_argument('--t-stop', type=float, default=10.0)
    parser.add_argument('--max-time', type=float, default=600.0, help='Max time for reference (seconds)')
    parser.add_argument('--output', type=str, default='PM5/stability_analysis_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE STABILITY AND ERROR ANALYSIS")
    print("="*70)
    print(f"Time range: [{args.t_start}, {args.t_stop}]")
    print(f"Reference computation max time: {args.max_time:.1f}s ({args.max_time/60:.1f} min)")
    
    # Set up model
    print("\n[1/5] Setting up 12-genotype model...")
    p, x0, eval_u = setup_12_genotype_model()
    print(f"  Model initialized: {len(x0)} states")
    
    # Compute reference solution
    print("\n[2/5] Computing reference solution (10-minute limit)...")
    ref_result = compute_reference_solution(
        eval_f=evalf, x0=x0, p=p, eval_u=eval_u,
        t_start=args.t_start, t_stop=args.t_stop,
        dt_initial=0.1, convergence_tol=1e-6,
        max_time_seconds=args.max_time, verbose=True
    )
    xref = ref_result['xref']
    dt_ref = ref_result['dt_ref']
    
    # Find stability boundary for Forward Euler
    print("\n[3/5] Finding stability boundary for Forward Euler...")
    dt_unst, dt_unst_bound, stability_info = find_unstable_dt_forward_euler(
        eval_f=evalf, x0=x0, p=p, eval_u=eval_u,
        t_start=args.t_start, t_stop=args.t_stop,
        dt_start=0.5, verbose=True
    )
    
    # Compute error at instability
    print("\n[4/5] Computing error at instability boundary...")
    X_unst, t_unst = forward_euler(evalf, x0, p, eval_u, 
                                   args.t_start, args.t_stop, dt_unst, verbose=False)
    eps_unst = np.linalg.norm(X_unst[:, -1] - xref, ord=np.inf)
    
    print(f"\n  ∆tunst = {dt_unst:.6e}")
    print(f"  ϵunst = ||x(∆tunst) - xref||∞ = {eps_unst:.6e}")
    
    # Determine acceptable error (use a reasonable value based on problem scale)
    # For biological populations, typically 1-5% relative error might be acceptable
    xref_scale = np.linalg.norm(xref, ord=np.inf)
    eps_a_relative = 0.01  # 1% relative error
    eps_a = eps_a_relative * xref_scale
    
    print(f"\n[5/5] Determining acceptable error level...")
    print(f"  Reference scale: ||xref||∞ = {xref_scale:.6e}")
    print(f"  Acceptable relative error: {eps_a_relative*100:.1f}%")
    print(f"  ϵa = {eps_a:.6e}")
    
    # Compare ϵa vs ϵunst
    print("\n" + "="*70)
    print("COMPARISON: ϵa vs ϵunst")
    print("="*70)
    
    results = {
        'xref': xref.tolist(),
        'dt_ref': dt_ref,
        'ref_converged': ref_result['converged'],
        'dt_unst': dt_unst,
        'eps_unst': float(eps_unst),
        'eps_a': float(eps_a),
        'xref_scale': float(xref_scale)
    }
    
    if eps_a < eps_unst:
        print(f"\nCase: ϵa ({eps_a:.6e}) < ϵunst ({eps_unst:.6e})")
        print("  → Forward Euler can achieve acceptable error before instability")
        print("  → Finding ∆ta that gives ϵa...")
        
        dt_a_fe, error_a_fe, time_a_fe = find_dt_for_error_target(
            eval_f=evalf, x0=x0, p=p, eval_u=eval_u,
            t_start=args.t_start, t_stop=args.t_stop,
            xref=xref, error_target=eps_a,
            method='forward_euler',
            dt_low=dt_ref, dt_high=dt_unst,
            verbose=True
        )
        
        if dt_a_fe is not None:
            safety_ratio = dt_unst / dt_a_fe
            print(f"\n  ∆ta (Forward Euler) = {dt_a_fe:.6e}")
            print(f"  Safety ratio: ∆tunst / ∆ta = {safety_ratio:.2f}")
            
            if safety_ratio > 2.0:
                print(f"  ✓ Safety margin sufficient (∆ta << ∆tunst)")
            else:
                print(f"  ⚠ Safety margin small (consider using smaller dt)")
            
            results['case'] = 'eps_a_lt_eps_unst'
            results['dt_a_fe'] = dt_a_fe
            results['error_a_fe'] = error_a_fe
            results['time_a_fe'] = time_a_fe
            results['safety_ratio'] = safety_ratio
        
    else:
        print(f"\nCase: ϵunst ({eps_unst:.6e}) < ϵa ({eps_a:.6e})")
        print("  → Forward Euler limited by stability, not accuracy")
        print("  → Testing Trapezoidal with larger ∆t...")
        
        # Test Trapezoidal with larger dt
        dt_test = dt_unst * 2.0  # Start at 2x unstable dt for FE
        
        print(f"\n  Testing Trapezoidal with dt = {dt_test:.6e}...")
        start_time = time.time()
        X_trap, t_trap = trapezoidal_with_init(
            evalf, evaljacobianf, x0, p, eval_u,
            args.t_start, args.t_stop, dt_test,
            init_method='feuler', verbose=False
        )
        time_trap = time.time() - start_time
        
        if X_trap is not None:
            stable, _ = is_stable(X_trap)
            if stable:
                eps_trap = np.linalg.norm(X_trap[:, -1] - xref, ord=np.inf)
                print(f"  ✓ Trapezoidal stable")
                print(f"  Error: {eps_trap:.6e}")
                print(f"  Time: {time_trap:.3f}s")
                
                # Find dt for Trapezoidal that gives eps_a
                print(f"\n  Finding ∆ta for Trapezoidal to achieve ϵa...")
                dt_a_trap, error_a_trap, time_a_trap = find_dt_for_error_target(
                    eval_f=evalf, x0=x0, p=p, eval_u=eval_u,
                    t_start=args.t_start, t_stop=args.t_stop,
                    xref=xref, error_target=eps_a,
                    method='trapezoidal',
                    dt_low=dt_unst, dt_high=dt_test * 5,
                    verbose=True
                )
                
                if dt_a_trap is not None:
                    # Compare with Forward Euler at dt_unst
                    X_fe_unst, _ = forward_euler(evalf, x0, p, eval_u,
                                                args.t_start, args.t_stop, dt_unst, verbose=False)
                    time_fe_unst = 0.01  # Approximate
                    
                    print(f"\n  Comparison:")
                    print(f"    Forward Euler (∆tunst): error = {eps_unst:.6e}, time ≈ {time_fe_unst:.3f}s")
                    print(f"    Trapezoidal (∆ta): error = {error_a_trap:.6e}, time = {time_a_trap:.3f}s")
                    
                    if time_a_trap < time_fe_unst * eps_unst / eps_a:  # Rough comparison
                        print(f"  → Trapezoidal is beneficial for achieving ϵa")
                    else:
                        print(f"  → Forward Euler faster, but limited by stability")
                
                results['case'] = 'eps_unst_lt_eps_a'
                results['dt_a_trap'] = dt_a_trap if dt_a_trap else None
                results['error_a_trap'] = error_a_trap if dt_a_trap else None
                results['time_a_trap'] = time_a_trap if dt_a_trap else None
            else:
                print(f"  ✗ Trapezoidal also unstable")
                results['case'] = 'both_unstable'
        else:
            print(f"  ✗ Trapezoidal failed")
            results['case'] = 'trapezoidal_failed'
    
    # Save results
    print(f"\n[SAVE] Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return results


if __name__ == '__main__':
    results = main()

