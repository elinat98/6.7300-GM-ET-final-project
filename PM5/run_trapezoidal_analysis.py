#!/usr/bin/env python3
"""
Run Trapezoidal comparisons and generate plots.

This script:
1. Compares initialization methods (previous vs Forward Euler prediction)
2. Compares fixed vs adaptive time-stepping
3. Generates plots
4. Updates LaTeX document
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from PM5.model_setup import setup_12_genotype_model
from PM5.stability_analysis import trapezoidal_fallback


def compare_init_methods_simple(eval_f, eval_Jf, x0, p, eval_u, t_start, t_stop, dt=0.03):
    """Simple comparison of initialization methods."""
    print("="*70)
    print("INITIALIZATION METHOD COMPARISON")
    print("="*70)
    print(f"dt = {dt:.6e}, t = [{t_start}, {t_stop}]")
    
    results = {}
    
    for init_method in ['previous', 'feuler']:
        print(f"\n[{'1' if init_method == 'previous' else '2'}/2] {init_method.upper()}")
        start_time = time.time()
        
        try:
            X, t, stats = trapezoidal_fallback(
                eval_f, eval_Jf, x0, p, eval_u,
                t_start, t_stop, dt,
                init_method=init_method, verbose=False
            )
            comp_time = time.time() - start_time
            
            if X is None:
                print(f"  ✗ Failed")
                results[init_method] = {'success': False}
                continue
            
            avg_iters = stats.get('avg_iterations', 0)
            max_iters = stats.get('max_iterations', 0)
            failures = stats.get('failures', 0)
            
            print(f"  ✓ Success")
            print(f"  Time: {comp_time:.4f} s")
            print(f"  Steps: {len(t)-1}")
            print(f"  Avg Newton iters: {avg_iters:.1f}")
            print(f"  Max Newton iters: {max_iters}")
            print(f"  Failures: {failures}")
            
            results[init_method] = {
                'success': True,
                'time': float(comp_time),
                'steps': len(t)-1,
                'avg_newton_iters': float(avg_iters),
                'max_newton_iters': int(max_iters),
                'failures': int(failures)
            }
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results[init_method] = {'success': False, 'error': str(e)}
    
    # Summary
    if results.get('previous', {}).get('success') and results.get('feuler', {}).get('success'):
        t_prev = results['previous']['time']
        t_fe = results['feuler']['time']
        speedup = t_prev / t_fe if t_fe > 0 else np.inf
        print(f"\n{'='*70}")
        print(f"Speedup: Forward Euler prediction is {speedup:.2f}x faster")
        print(f"  Previous: {t_prev:.4f} s")
        print(f"  FEuler: {t_fe:.4f} s")
    
    return results


def run_fixed_trapezoidal(eval_f, eval_Jf, x0, p, eval_u, t_start, t_stop, dt, xref):
    """Run fixed Trapezoidal and compute error."""
    start_time = time.time()
    X, t, stats = trapezoidal_fallback(
        eval_f, eval_Jf, x0, p, eval_u,
        t_start, t_stop, dt,
        init_method='feuler', verbose=False
    )
    comp_time = time.time() - start_time
    
    if X is None:
        return None
    
    error = np.linalg.norm(X[:, -1] - xref, ord=np.inf)
    
    return {
        'X': X,
        't': t,
        'time': float(comp_time),
        'steps': len(t) - 1,
        'error': float(error),
        'dt': float(dt),
        'newton_stats': {
            'avg_iterations': float(stats.get('avg_iterations', 0)),
            'max_iterations': int(stats.get('max_iterations', 0)),
            'failures': int(stats.get('failures', 0))
        }
    }


def main():
    """Run analysis."""
    print("="*70)
    print("TRAPEZOIDAL METHOD ANALYSIS")
    print("="*70)
    
    # Setup
    p, x0, eval_u = setup_12_genotype_model()
    t_start, t_stop = 0.0, 2.0  # Shorter for faster testing
    
    # Load reference
    with open('PM5/stability_results.json', 'r') as f:
        ref_data = json.load(f)
    xref = np.array(ref_data['xref'])
    error_target = ref_data['eps_a']
    
    # Comparison 1: Initialization
    print("\n" + "="*70)
    init_results = compare_init_methods_simple(
        evalf, evaljacobianf, x0, p, eval_u,
        t_start, t_stop, dt=0.03
    )
    
    # Save
    with open('PM5/initialization_results.json', 'w') as f:
        json.dump(init_results, f, indent=2)
    
    # Comparison 2: Fixed Trapezoidal at different dt
    print("\n" + "="*70)
    print("FIXED TRAPEZOIDAL: FINDING dt FOR ERROR TARGET")
    print("="*70)
    
    dt_values = [0.2, 0.15, 0.1, 0.05, 0.03]
    fixed_results = []
    
    for dt in dt_values:
        print(f"\nTesting dt = {dt:.4f}...")
        result = run_fixed_trapezoidal(
            evalf, evaljacobianf, x0, p, eval_u,
            t_start, t_stop, dt, xref
        )
        if result:
            print(f"  Error: {result['error']:.6e}, Time: {result['time']:.4f} s")
            fixed_results.append(result)
    
    # Find dt closest to error_target
    if fixed_results:
        best = min(fixed_results, key=lambda r: abs(r['error'] - error_target))
        print(f"\n{'='*70}")
        print(f"Best fixed dt: {best['dt']:.6e}")
        print(f"  Error: {best['error']:.6e} (target: {error_target:.6e})")
        print(f"  Time: {best['time']:.4f} s")
        print(f"  Steps: {best['steps']}")
    
    # Save
    with open('PM5/fixed_trapezoidal_results.json', 'w') as f:
        # Convert to JSON-serializable
        json_data = []
        for r in fixed_results:
            json_data.append({
                'dt': r['dt'],
                'time': r['time'],
                'steps': r['steps'],
                'error': r['error'],
                'newton_stats': r['newton_stats']
            })
        json.dump(json_data, f, indent=2)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nResults saved:")
    print("  - PM5/initialization_results.json")
    print("  - PM5/fixed_trapezoidal_results.json")
    
    return init_results, fixed_results


if __name__ == '__main__':
    init_results, fixed_results = main()

