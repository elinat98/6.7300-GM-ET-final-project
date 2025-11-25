#!/usr/bin/env python3
"""
Compare explicit vs implicit ODE time integrators for 12-genotype bacterial model.

This script:
1. Runs Forward Euler with different fixed ∆t values
2. Computes a reference solution by convergence testing
3. Compares results (will be extended for implicit methods)

For now, focuses on Forward Euler and reference solution computation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalf_bacterial import evalf
from PM5.forward_euler import forward_euler
from PM5.reference_solution import compute_reference_solution
from PM5.model_setup import setup_12_genotype_model


def compare_forward_euler_dt(p, x0, eval_u, t_start=0.0, t_stop=10.0,
                             dt_values=None, xref=None, dt_ref=None,
                             verbose=True):
    """
    Run Forward Euler for different dt values and compare to reference.
    
    Parameters
    ----------
    p : dict
        Model parameters
    x0 : array
        Initial state
    eval_u : callable
        Input function
    t_start : float
        Starting time
    t_stop : float
        Stopping time
    dt_values : list of float
        Time step values to test (if None, uses default range)
    xref : array, optional
        Reference solution at t_stop (if None, will be computed)
    dt_ref : float, optional
        Reference time step (for display only)
    verbose : bool
        If True, print results
    
    Returns
    -------
    results : dict
        Dictionary with dt_values, errors, computation times, etc.
    """
    
    if dt_values is None:
        # Default range: from 0.1 down to 0.001
        dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    errors = []
    computation_times = []
    final_states = []
    
    if verbose:
        print("="*70)
        print("FORWARD EULER COMPARISON: DIFFERENT ∆t VALUES")
        print("="*70)
        if xref is not None:
            print(f"Reference solution available (dt_ref = {dt_ref:.6e})")
        print(f"Time range: [{t_start}, {t_stop}]")
        print(f"Testing {len(dt_values)} different dt values")
        print("-"*70)
    
    for i, dt in enumerate(dt_values):
        if verbose:
            print(f"\n[{i+1}/{len(dt_values)}] Running Forward Euler with dt = {dt:.6e}")
        
        start_time = time.time()
        try:
            X, t = forward_euler(eval_f=evalf, x0=x0, p=p, eval_u=eval_u,
                                t_start=t_start, t_stop=t_stop, dt=dt,
                                verbose=False)
            x_final = X[:, -1]
            comp_time = time.time() - start_time
            
            final_states.append(x_final)
            
            # Compute error if reference is available
            if xref is not None:
                error = np.linalg.norm(x_final - xref, ord=np.inf)
                errors.append(error)
                
                if verbose:
                    print(f"  Computation time: {comp_time:.2f}s")
                    print(f"  Error ||x_final - x_ref||∞ = {error:.6e}")
                    print(f"  Final state norm: {np.linalg.norm(x_final):.6e}")
            else:
                if verbose:
                    print(f"  Computation time: {comp_time:.2f}s")
                    print(f"  Final state norm: {np.linalg.norm(x_final):.6e}")
            
            computation_times.append(comp_time)
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            errors.append(np.nan)
            computation_times.append(np.nan)
            final_states.append(None)
    
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'dt':<12} {'Time (s)':<12} {'Error (∞-norm)':<20}")
        print("-"*70)
        for dt, comp_time, error in zip(dt_values, computation_times, errors):
            if np.isnan(error):
                print(f"{dt:<12.6e} {'FAILED':<12} {'N/A':<20}")
            else:
                print(f"{dt:<12.6e} {comp_time:<12.2f} {error:<20.6e}")
    
    return {
        'dt_values': dt_values,
        'errors': errors,
        'computation_times': computation_times,
        'final_states': final_states,
        'xref': xref,
        'dt_ref': dt_ref
    }


def main():
    """Main function to run integrator comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare explicit vs implicit ODE integrators for 12-genotype model'
    )
    parser.add_argument('--t-start', type=float, default=0.0,
                       help='Starting time (default: 0.0)')
    parser.add_argument('--t-stop', type=float, default=10.0,
                       help='Stopping time (default: 10.0)')
    parser.add_argument('--dt-initial', type=float, default=0.1,
                       help='Initial dt for reference solution (default: 0.1)')
    parser.add_argument('--convergence-tol', type=float, default=1e-6,
                       help='Convergence tolerance for reference (default: 1e-6)')
    parser.add_argument('--max-time', type=float, default=300.0,
                       help='Max computation time in seconds (default: 300.0 = 5 min)')
    parser.add_argument('--skip-reference', action='store_true',
                       help='Skip reference solution computation')
    parser.add_argument('--dt-values', type=str, default=None,
                       help='Comma-separated dt values to test (default: auto)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ODE INTEGRATOR COMPARISON: 12-GENOTYPE BACTERIAL MODEL")
    print("="*70)
    
    # Set up model
    print("\n[1/3] Setting up 12-genotype model...")
    p, x0, eval_u = setup_12_genotype_model(
        mutation_type='fitness_landscape',
        uR=0.5,
        uC=0.1
    )
    m = 12
    print(f"  Model: {m} genotypes + resource + antibiotic = {len(x0)} states")
    print(f"  Initial state norm: {np.linalg.norm(x0):.4f}")
    
    # Compute reference solution
    xref = None
    dt_ref = None
    ref_result = None
    
    if not args.skip_reference:
        print("\n[2/3] Computing reference solution...")
        ref_result = compute_reference_solution(
            eval_f=evalf,
            x0=x0,
            p=p,
            eval_u=eval_u,
            t_start=args.t_start,
            t_stop=args.t_stop,
            dt_initial=args.dt_initial,
            convergence_tol=args.convergence_tol,
            max_time_seconds=args.max_time,
            verbose=True
        )
        xref = ref_result['xref']
        dt_ref = ref_result['dt_ref']
        print(f"\nReference solution:")
        print(f"  Converged: {ref_result['converged']}")
        print(f"  dt_ref: {dt_ref:.6e}")
        print(f"  Iterations: {ref_result['iterations']}")
        print(f"  Time elapsed: {ref_result['time_elapsed']:.2f}s")
    else:
        print("\n[2/3] Skipping reference solution computation")
    
    # Run Forward Euler for different dt values
    print("\n[3/3] Running Forward Euler for different dt values...")
    
    dt_values = None
    if args.dt_values is not None:
        dt_values = [float(x.strip()) for x in args.dt_values.split(',')]
    
    results = compare_forward_euler_dt(
        p=p, x0=x0, eval_u=eval_u,
        t_start=args.t_start, t_stop=args.t_stop,
        dt_values=dt_values,
        xref=xref, dt_ref=dt_ref,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nReady for next steps (implicit integrators, visualization, etc.)")
    
    return results, ref_result


if __name__ == '__main__':
    results, ref_result = main()

