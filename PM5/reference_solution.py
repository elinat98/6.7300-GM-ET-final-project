"""
Compute reference solution by convergence testing.

Progressively reduces time step ∆t until convergence:
||x(tstop)_∆ti - x(tstop)_∆ti-1||∞ < εref

Or stops after 5 minutes of computation.
"""

import sys
from pathlib import Path
import numpy as np
import time
from typing import Callable, Tuple, Dict, Optional

# Ensure we can import forward_euler
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PM5.forward_euler import forward_euler


def compute_reference_solution(eval_f: Callable,
                               x0: np.ndarray,
                               p: dict,
                               eval_u: Callable,
                               t_start: float,
                               t_stop: float,
                               dt_initial: float = 0.1,
                               convergence_tol: float = 1e-6,
                               max_time_seconds: float = 300.0,  # 5 minutes
                               dt_reduction_factor: float = 2.0,
                               verbose: bool = True) -> Dict:
    """
    Compute reference solution by progressively refining time step.
    
    Parameters
    ----------
    eval_f : callable
        Function that evaluates f(x, p, u)
    x0 : array
        Initial state vector
    p : dict
        Model parameters
    eval_u : callable
        Input function u(t)
    t_start : float
        Starting time
    t_stop : float
        Stopping time
    dt_initial : float
        Initial time step (default: 0.1)
    convergence_tol : float
        Convergence tolerance εref (default: 1e-6)
    max_time_seconds : float
        Maximum computation time in seconds (default: 300 = 5 min)
    dt_reduction_factor : float
        Factor to reduce dt by each iteration (default: 2.0)
    verbose : bool
        If True, print progress information
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'xref': reference solution at final time, shape (N,)
        - 'dt_ref': reference time step
        - 'converged': bool, whether convergence was achieved
        - 'time_elapsed': float, computation time in seconds
        - 'iterations': int, number of refinement iterations
        - 'convergence_history': list of (dt, error, time) tuples
    """
    
    start_time = time.time()
    
    dt_current = dt_initial
    x_prev = None
    convergence_history = []
    iteration = 0
    
    if verbose:
        print("="*70)
        print("COMPUTING REFERENCE SOLUTION")
        print("="*70)
        print(f"Time range: [{t_start}, {t_stop}]")
        print(f"Convergence tolerance: εref = {convergence_tol:.2e}")
        print(f"Max computation time: {max_time_seconds:.1f} seconds")
        print("-"*70)
    
    while True:
        iteration += 1
        iter_start_time = time.time()
        
        if verbose:
            print(f"\nIteration {iteration}: dt = {dt_current:.6e}")
        
        # Run Forward Euler with current dt
        try:
            X, t = forward_euler(
                eval_f, x0, p, eval_u, t_start, t_stop, dt_current,
                verbose=False
            )
            x_current = X[:, -1]  # Final state
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            break
        
        # Compute error if we have previous solution
        if x_prev is not None:
            error = np.linalg.norm(x_current - x_prev, ord=np.inf)
            iter_time = time.time() - iter_start_time
            total_time = time.time() - start_time
            
            convergence_history.append((dt_current, error, iter_time))
            
            if verbose:
                print(f"  Error: ||x_∆ti - x_∆ti-1||∞ = {error:.6e}")
                print(f"  Iteration time: {iter_time:.2f}s, Total: {total_time:.2f}s")
            
            # Check convergence
            if error < convergence_tol:
                if verbose:
                    print(f"\n✓ CONVERGED after {iteration} iterations")
                    print(f"  Final dt: {dt_current:.6e}")
                    print(f"  Final error: {error:.6e} < {convergence_tol:.2e}")
                    print(f"  Total time: {total_time:.2f}s")
                
                return {
                    'xref': x_current,
                    'dt_ref': dt_current,
                    'converged': True,
                    'time_elapsed': total_time,
                    'iterations': iteration,
                    'convergence_history': convergence_history
                }
            
            # Check time limit (check BEFORE next iteration to avoid going over)
            if total_time >= max_time_seconds:
                if verbose:
                    print(f"\n⏱ TIME LIMIT REACHED ({max_time_seconds:.1f}s)")
                    print(f"  Final dt: {dt_current:.6e}")
                    print(f"  Final error: {error:.6e} > {convergence_tol:.2e}")
                    print(f"  Stopping - using best solution so far")
                
                return {
                    'xref': x_current,
                    'dt_ref': dt_current,
                    'converged': False,
                    'time_elapsed': total_time,
                    'iterations': iteration,
                    'convergence_history': convergence_history
                }
        else:
            iter_time = time.time() - iter_start_time
            total_time = time.time() - start_time
            convergence_history.append((dt_current, np.inf, iter_time))
            
            if verbose:
                print(f"  Iteration time: {iter_time:.2f}s, Total: {total_time:.2f}s")
        
        # Prepare for next iteration
        x_prev = x_current.copy()
        dt_current = dt_current / dt_reduction_factor
        
        # Safety check: avoid extremely small dt
        if dt_current < 1e-10:
            if verbose:
                print(f"\n⚠ dt too small ({dt_current:.6e}), stopping")
                print(f"  Final error: {error:.6e}")
            
            return {
                'xref': x_current,
                'dt_ref': dt_current * dt_reduction_factor,  # Use previous dt
                'converged': False,
                'time_elapsed': time.time() - start_time,
                'iterations': iteration,
                'convergence_history': convergence_history
            }
    
    # Fallback return
    return {
        'xref': x_prev if x_prev is not None else x0,
        'dt_ref': dt_current * dt_reduction_factor,
        'converged': False,
        'time_elapsed': time.time() - start_time,
        'iterations': iteration,
        'convergence_history': convergence_history
    }

