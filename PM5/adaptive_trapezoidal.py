"""
Adaptive Trapezoidal integrator with dynamic time-stepping.

Adjusts time step based on:
1. Solution variation rate (derivative estimation)
2. Error monitoring (if reference solution available)
"""

import numpy as np
import time
from typing import Callable, Tuple, Dict, Optional
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PM4.newton.NewtonSolver import newton_solve


def adaptive_trapezoidal(eval_f: Callable,
                         eval_Jf: Callable,
                         x0: np.ndarray,
                         p: dict,
                         eval_u: Callable,
                         t_start: float,
                         t_stop: float,
                         dt_initial: float = 0.1,
                         dt_min: float = 1e-6,
                         dt_max: float = 1.0,
                         error_target: Optional[float] = None,
                         xref: Optional[np.ndarray] = None,
                         safety_factor: float = 0.9,
                         growth_factor: float = 1.5,
                         shrink_factor: float = 0.5,
                         variation_tol: float = 0.1,
                         max_iter_per_step: int = 50,
                         verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Adaptive Trapezoidal integrator with dynamic time-stepping.
    
    Parameters
    ----------
    eval_f : callable
        ODE right-hand side f(x, p, u)
    eval_Jf : callable
        Jacobian J_f(x, p, u)
    x0 : array
        Initial state
    p : dict
        Model parameters
    eval_u : callable
        Input function u(t)
    t_start, t_stop : float
        Time range
    dt_initial : float
        Initial time step
    dt_min, dt_max : float
        Minimum and maximum allowed time steps
    error_target : float, optional
        Target error level ϵa (if monitoring error)
    xref : array, optional
        Reference solution (for error monitoring)
    safety_factor : float
        Safety factor for time step adjustment (default: 0.9)
    growth_factor : float
        Factor to grow dt when solution is smooth (default: 1.5)
    shrink_factor : float
        Factor to shrink dt when solution changes quickly (default: 0.5)
    variation_tol : float
        Tolerance for solution variation rate (default: 0.1)
    max_iter_per_step : int
        Maximum Newton iterations per step
    verbose : bool
        If True, print progress
    
    Returns
    -------
    X : array
        State trajectory
    t : array
        Time points
    info : dict
        Statistics (number of steps, rejections, etc.)
    """
    
    x0_flat = np.asarray(x0, dtype=float).ravel()
    N = len(x0_flat)
    
    # Storage for trajectory (will grow dynamically)
    max_steps_estimate = int(np.ceil((t_stop - t_start) / dt_min)) + 1000
    X = np.zeros((N, max_steps_estimate))
    t = np.zeros(max_steps_estimate)
    
    X[:, 0] = x0_flat
    t[0] = t_start
    n = 0
    dt_current = dt_initial
    
    # Statistics
    stats = {
        'steps_accepted': 0,
        'steps_rejected': 0,
        'dt_values': [],
        'errors': [],
        'variation_rates': [],
        'newton_iterations': []
    }
    
    if verbose:
        print(f"Adaptive Trapezoidal: t = [{t_start}, {t_stop}]")
        print(f"  Initial dt = {dt_initial:.6e}")
        print(f"  dt range: [{dt_min:.6e}, {dt_max:.6e}]")
        if error_target is not None:
            print(f"  Error target: {error_target:.6e}")
    
    while t[n] < t_stop:
        # Ensure we don't go past t_stop
        dt_actual = min(dt_current, t_stop - t[n])
        
        if dt_actual < dt_min:
            dt_actual = t_stop - t[n]
            if dt_actual < dt_min * 0.1:  # Very small step remaining
                break
        
        x_n = X[:, n].reshape((-1, 1))
        u_n = eval_u(t[n])
        u_np1 = eval_u(t[n] + dt_actual)
        f_n = eval_f(x_n, p, u_n)
        
        # Define residual for Trapezoidal
        # Newton solver expects: residual(x, p, u)
        def residual(x_new, p_dummy, u_dummy):
            x_new_col = x_new.reshape((-1, 1)) if x_new.ndim == 1 else x_new
            f_new = eval_f(x_new_col, p, u_np1)
            f_n_col = f_n.reshape((-1, 1)) if f_n.ndim == 1 else f_n
            x_n_col = x_n.reshape((-1, 1)) if x_n.ndim == 1 else x_n
            return x_new_col - x_n_col - (dt_actual / 2.0) * (f_n_col + f_new)
        
        # Jacobian of residual
        # Newton solver expects: jacobian(x, p, u)
        def residual_jacobian(x_new, p_dummy, u_dummy):
            x_new_col = x_new.reshape((-1, 1)) if x_new.ndim == 1 else x_new
            I = np.eye(N)
            J_f = eval_Jf(x_new_col, p, u_np1)
            return I - (dt_actual / 2.0) * J_f
        
        # Initial guess: Forward Euler prediction
        x_guess = (x_n + dt_actual * f_n.ravel()).reshape((-1, 1))
        
        # Solve with Newton
        try:
            x_new, info = newton_solve(
                residual, x_guess, {}, np.zeros(0),
                jacobian_fn=residual_jacobian,
                max_iter=max_iter_per_step,
                tol=1e-10,
                verbose=False
            )
            
            if not info['converged']:
                # Newton failed - reject step and shrink
                stats['steps_rejected'] += 1
                dt_current *= shrink_factor
                dt_current = max(dt_current, dt_min)
                if verbose:
                    print(f"  Newton failed at t = {t[n]:.4f}, shrinking dt to {dt_current:.6e}")
                continue
            
            newton_iters = info['iters']
            stats['newton_iterations'].append(newton_iters)
        except Exception as e:
            # Newton solve exception - reject step and shrink
            stats['steps_rejected'] += 1
            dt_current *= shrink_factor
            dt_current = max(dt_current, dt_min)
            if verbose:
                print(f"  Newton exception at t = {t[n]:.4f}: {e}, shrinking dt to {dt_current:.6e}")
            continue
            
            # Check if we need to resize storage
            if n + 1 >= max_steps_estimate:
                # Double storage
                X_new = np.zeros((N, max_steps_estimate * 2))
                t_new = np.zeros(max_steps_estimate * 2)
                X_new[:, :n+1] = X[:, :n+1]
                t_new[:n+1] = t[:n+1]
                X, t = X_new, t_new
                max_steps_estimate *= 2
            
            # Ensure x_new is properly shaped
            x_new_flat = np.asarray(x_new, dtype=float).ravel()
            if len(x_new_flat) != N:
                # This shouldn't happen, but handle it
                stats['steps_rejected'] += 1
                dt_current *= shrink_factor
                dt_current = max(dt_current, dt_min)
                if verbose:
                    print(f"  State dimension mismatch at step {n+1}, shrinking dt")
                continue
            
            # Compute metrics to decide if step should be accepted
            x_n_flat = x_n.ravel()
            
            # 1. Solution variation rate
            dx_dt_approx = (x_new_flat - x_n_flat) / dt_actual
            variation_rate = np.linalg.norm(dx_dt_approx, ord=np.inf)
            stats['variation_rates'].append(variation_rate)
            
            # 2. Error monitoring (if reference available)
            current_error = None
            if xref is not None:
                current_error = np.linalg.norm(x_new_flat - xref, ord=np.inf)
                stats['errors'].append(current_error)
            
            # Acceptance criteria
            accept_step = True
            
            # Criterion 1: Variation rate too high → reject and shrink
            if variation_rate > 10.0:  # Heuristic threshold
                accept_step = False
                reason = "variation rate too high"
            
            # Criterion 2: Error too high (if monitoring)
            if error_target is not None and current_error is not None:
                if current_error > error_target * 2.0:  # 2x tolerance
                    accept_step = False
                    reason = "error too high"
            
            # Criterion 3: Solution explosion
            if np.max(np.abs(x_new_flat)) > 1e10:
                accept_step = False
                reason = "solution explosion"
            
            if not accept_step:
                stats['steps_rejected'] += 1
                dt_current *= shrink_factor
                dt_current = max(dt_current, dt_min)
                if verbose:
                    print(f"  Rejected at t = {t[n]:.4f}: {reason}, dt → {dt_current:.6e}")
                continue
            
            # Step accepted
            n += 1
            X[:, n] = x_new_flat
            t[n] = t[n-1] + dt_actual
            stats['steps_accepted'] += 1
            stats['dt_values'].append(dt_actual)
            
            # Adjust time step for next iteration
            
            # 1. Based on variation rate
            if variation_rate > variation_tol:
                # Solution changing quickly → shrink dt
                dt_next = dt_current * shrink_factor
            elif variation_rate < variation_tol * 0.1:
                # Solution smooth → grow dt
                dt_next = dt_current * growth_factor
            else:
                # Moderate variation → keep similar dt
                dt_next = dt_current
            
            # 2. Based on error (if monitoring)
            if error_target is not None and current_error is not None:
                error_ratio = current_error / error_target if error_target > 0 else 1.0
                if error_ratio > 1.5:
                    # Error too high → shrink
                    dt_next = min(dt_next, dt_current * shrink_factor)
                elif error_ratio < 0.5:
                    # Error very low → can grow
                    dt_next = max(dt_next, dt_current * growth_factor)
            
            # 3. Based on Newton iterations
            if newton_iters > max_iter_per_step * 0.8:
                # Many iterations → shrink dt for easier convergence
                dt_next = min(dt_next, dt_current * shrink_factor)
            elif newton_iters < 3:
                # Few iterations → can grow
                dt_next = max(dt_next, dt_current * growth_factor)
            
            # Apply safety factor and bounds
            dt_current = safety_factor * dt_next
            dt_current = max(dt_min, min(dt_max, dt_current))
            
            if verbose and n % max(1, int(100 / dt_initial)) == 0:
                print(f"  t = {t[n]:.4f}, dt = {dt_current:.6e}, steps = {n}, "
                      f"var_rate = {variation_rate:.4e}", end="")
                if current_error is not None:
                    print(f", error = {current_error:.6e}")
                else:
                    print()
    
    # Trim arrays to actual size
    X = X[:, :n+1]
    t = t[:n+1]
    
    if verbose:
        print(f"\nAdaptive Trapezoidal complete:")
        print(f"  Final t = {t[-1]:.4f}")
        print(f"  Steps accepted: {stats['steps_accepted']}")
        print(f"  Steps rejected: {stats['steps_rejected']}")
        print(f"  Avg dt: {np.mean(stats['dt_values']) if stats['dt_values'] else 0:.6e}")
        print(f"  Min dt: {np.min(stats['dt_values']) if stats['dt_values'] else 0:.6e}")
        print(f"  Max dt: {np.max(stats['dt_values']) if stats['dt_values'] else 0:.6e}")
        if stats['errors']:
            print(f"  Final error: {stats['errors'][-1]:.6e}")
    
    return X, t, stats


def compare_fixed_vs_adaptive(eval_f, eval_Jf, x0, p, eval_u, t_start, t_stop,
                               xref, error_target, dt_fixed=None, verbose=True):
    """
    Compare fixed vs adaptive time-stepping for Trapezoidal method.
    
    Parameters
    ----------
    dt_fixed : float, optional
        Fixed time step to use. If None, uses dt that gives error_target.
    
    Returns
    -------
    results : dict
        Comparison results
    """
    
    if verbose:
        print("="*70)
        print("COMPARING FIXED vs ADAPTIVE TIME-STEPPING (Trapezoidal)")
        print("="*70)
    
    # 1. Find fixed dt that gives approximately error_target
    if dt_fixed is None:
        if verbose:
            print("\n[1/3] Finding fixed dt for error target...")
        
        # Binary search for fixed dt
        dt_low = 1e-3
        dt_high = 1.0
        dt_fixed = 0.1  # Initial guess
        
        for i in range(20):
            dt_test = np.sqrt(dt_low * dt_high)
            
            # Run fixed Trapezoidal
            from PM5.stability_analysis import trapezoidal_fallback
            start_time = time.time()
            X_fixed, t_fixed, _ = trapezoidal_fallback(
                eval_f, eval_Jf, x0, p, eval_u,
                t_start, t_stop, dt_test,
                init_method='feuler', verbose=False
            )
            time_fixed = time.time() - start_time
            
            if X_fixed is None:
                dt_high = dt_test
                continue
            
            error_fixed = np.linalg.norm(X_fixed[:, -1] - xref, ord=np.inf)
            
            if abs(error_fixed - error_target) / error_target < 0.2:  # Within 20%
                dt_fixed = dt_test
                break
            
            if error_fixed > error_target:
                dt_high = dt_test
            else:
                dt_low = dt_test
        
        if verbose:
            print(f"  Fixed dt: {dt_fixed:.6e}")
    
    # 2. Run fixed Trapezoidal
    if verbose:
        print("\n[2/3] Running fixed Trapezoidal...")
    
    from PM5.stability_analysis import trapezoidal_fallback
    start_time = time.time()
    X_fixed, t_fixed, _ = trapezoidal_fallback(
        eval_f, eval_Jf, x0, p, eval_u,
        t_start, t_stop, dt_fixed,
        init_method='feuler', verbose=False
    )
    time_fixed = time.time() - start_time
    
    if X_fixed is None:
        if verbose:
            print("  Fixed Trapezoidal failed")
        return None
    
    error_fixed = np.linalg.norm(X_fixed[:, -1] - xref, ord=np.inf)
    steps_fixed = len(t_fixed) - 1
    
    if verbose:
        print(f"  Time: {time_fixed:.4f} seconds")
        print(f"  Steps: {steps_fixed}")
        print(f"  Error: {error_fixed:.6e}")
    
    # 3. Run adaptive Trapezoidal
    if verbose:
        print("\n[3/3] Running adaptive Trapezoidal...")
    
    start_time = time.time()
    X_adaptive, t_adaptive, stats_adaptive = adaptive_trapezoidal(
        eval_f, eval_Jf, x0, p, eval_u,
        t_start, t_stop,
        dt_initial=dt_fixed,
        dt_min=1e-5,
        dt_max=1.0,
        error_target=error_target,
        xref=xref,
        verbose=verbose
    )
    time_adaptive = time.time() - start_time
    
    error_adaptive = np.linalg.norm(X_adaptive[:, -1] - xref, ord=np.inf)
    steps_adaptive = stats_adaptive['steps_accepted']
    
    if verbose:
        print(f"  Time: {time_adaptive:.4f} seconds")
        print(f"  Steps: {steps_adaptive}")
        print(f"  Error: {error_adaptive:.6e}")
        print(f"  Rejections: {stats_adaptive['steps_rejected']}")
    
    # Comparison
    speedup = time_fixed / time_adaptive if time_adaptive > 0 else np.inf
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"Fixed Trapezoidal:")
        print(f"  Time: {time_fixed:.4f} s")
        print(f"  Steps: {steps_fixed}")
        print(f"  Error: {error_fixed:.6e}")
        print(f"  Avg dt: {dt_fixed:.6e}")
        print(f"\nAdaptive Trapezoidal:")
        print(f"  Time: {time_adaptive:.4f} s")
        print(f"  Steps: {steps_adaptive}")
        print(f"  Rejections: {stats_adaptive['steps_rejected']}")
        print(f"  Error: {error_adaptive:.6e}")
        print(f"  Avg dt: {np.mean(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0:.6e}")
        print(f"  Min dt: {np.min(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0:.6e}")
        print(f"  Max dt: {np.max(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0:.6e}")
        print(f"\nSpeedup: {speedup:.2f}x ({'adaptive faster' if speedup > 1 else 'adaptive slower'})")
    
    return {
        'fixed': {
            'dt': dt_fixed,
            'time': time_fixed,
            'steps': steps_fixed,
            'error': error_fixed
        },
        'adaptive': {
            'time': time_adaptive,
            'steps': steps_adaptive,
            'rejections': stats_adaptive['steps_rejected'],
            'error': error_adaptive,
            'dt_avg': np.mean(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0,
            'dt_min': np.min(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0,
            'dt_max': np.max(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0,
            'stats': stats_adaptive
        },
        'speedup': speedup
    }


if __name__ == '__main__':
    # Test the adaptive integrator
    print("Adaptive Trapezoidal integrator implementation")

