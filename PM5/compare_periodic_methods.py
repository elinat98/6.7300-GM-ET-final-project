#!/usr/bin/env python3
"""
Compare methods for finding periodic steady state:
1. Shooting-Newton
2. Forward Euler transient
3. Trapezoidal transient

Compare computation time for same error level.
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
from PM5.forward_euler import forward_euler
from PM5.stability_analysis import trapezoidal_fallback
from PM5.periodic_steady_state import (
    create_periodic_input, 
    state_transition,
    periodic_steady_state_solver,
    check_periodic_convergence
)


def find_periodic_ref_solution(evalf, eval_Jf, x0_guess, p, eval_u, period, dt_ref=0.001):
    """Find reference periodic steady state with very small dt."""
    print(f"Computing reference periodic steady state (dt={dt_ref:.4f})...")
    
    # Run transient for initial guess
    X_transient, t_transient = forward_euler(
        evalf, x0_guess, p, eval_u,
        0.0, 30.0 * period, dt_ref, verbose=False
    )
    x0_ref_guess = X_transient[:, -1]
    
    # Solve with Shooting-Newton using small dt
    x_periodic_ref, info = periodic_steady_state_solver(
        evalf, eval_Jf, x0_ref_guess, p, eval_u,
        period, dt_ref, tol=1e-10, max_iter=50, verbose=False
    )
    
    if not info['converged']:
        print("Warning: Reference solution did not converge. Using best guess.")
    
    # Verify
    X_verify, _ = forward_euler(
        evalf, x_periodic_ref, p, eval_u,
        0.0, period, dt_ref, verbose=False
    )
    error_ref = np.linalg.norm(X_verify[:, -1] - x_periodic_ref, ord=np.inf)
    
    print(f"  Reference verification error: {error_ref:.6e}")
    
    return x_periodic_ref, dt_ref, error_ref


def compute_periodic_error(x_periodic, x_periodic_ref):
    """Compute error of periodic solution relative to reference."""
    return np.linalg.norm(x_periodic - x_periodic_ref, ord=np.inf)


def transient_forward_euler_to_periodic(evalf, x0, p, eval_u, period, dt, 
                                         x_periodic_ref, error_target, max_periods=200, verbose=True):
    """
    Run Forward Euler transient until periodic steady state is reached.
    
    Returns:
    -------
    results : dict with 'time', 'periods', 'error', 'X', 't'
    """
    start_time = time.time()
    
    # Run in chunks to check convergence periodically
    chunk_periods = 20  # Larger chunks for efficiency
    total_periods = 0
    X_all = None
    t_all = None
    
    while total_periods < max_periods:
        if verbose and total_periods % 20 == 0:
            print(f"    Running period {total_periods}...")
        # Run next chunk
        t_start_chunk = total_periods * period
        t_stop_chunk = (total_periods + chunk_periods) * period
        
        X_chunk, t_chunk = forward_euler(
            evalf, x0, p, eval_u,
            t_start_chunk, t_stop_chunk, dt, verbose=False
        )
        
        # Concatenate
        if X_all is None:
            X_all = X_chunk
            t_all = t_chunk
        else:
            X_all = np.hstack([X_all, X_chunk[:, 1:]])  # Skip duplicate first point
            t_all = np.concatenate([t_all, t_chunk[1:]])
        
        # Update initial condition for next chunk
        x0 = X_chunk[:, -1]
        total_periods += chunk_periods
        
        # Check convergence
        converged, conv_period, max_diff = check_periodic_convergence(
            X_all, t_all, period, total_periods / period, tol=1e-3
        )
        
        # Check error relative to reference
        # Use final state as periodic solution estimate
        x_periodic_est = X_all[:, -1]
        error = compute_periodic_error(x_periodic_est, x_periodic_ref)
        
        # Check if we've reached error target (if specified)
        if error_target is not None and error < error_target:
            comp_time = time.time() - start_time
            return {
                'method': 'Forward Euler',
                'converged': True,
                'time': comp_time,
                'periods': total_periods,
                'error': error,
                'dt': dt,
                'X': X_all,
                't': t_all
            }
        
        if converged:
            comp_time = time.time() - start_time
            return {
                'method': 'Forward Euler',
                'converged': True,
                'time': comp_time,
                'periods': total_periods,
                'error': error,
                'dt': dt,
                'X': X_all,
                't': t_all
            }
    
    comp_time = time.time() - start_time
    # Compute final error if not already computed
    if X_all is not None:
        x_periodic_est = X_all[:, -1]
        error = compute_periodic_error(x_periodic_est, x_periodic_ref)
    else:
        error = np.inf
    
    return {
        'method': 'Forward Euler',
        'converged': False,
        'time': comp_time,
        'periods': total_periods,
        'error': error,
        'dt': dt,
        'X': X_all,
        't': t_all
    }


def transient_trapezoidal_to_periodic(evalf, eval_Jf, x0, p, eval_u, period, dt,
                                       x_periodic_ref, error_target, max_periods=200, verbose=True):
    """
    Run Trapezoidal transient until periodic steady state is reached.
    """
    start_time = time.time()
    
    chunk_periods = 20  # Larger chunks for efficiency
    total_periods = 0
    X_all = None
    t_all = None
    
    while total_periods < max_periods:
        if verbose and total_periods % 20 == 0:
            print(f"    Running period {total_periods}...")
        t_start_chunk = total_periods * period
        t_stop_chunk = (total_periods + chunk_periods) * period
        
        X_chunk, t_chunk, _ = trapezoidal_fallback(
            evalf, eval_Jf, x0, p, eval_u,
            t_start_chunk, t_stop_chunk, dt,
            init_method='feuler', verbose=False
        )
        
        if X_chunk is None:
            return {
                'method': 'Trapezoidal',
                'converged': False,
                'time': time.time() - start_time,
                'periods': total_periods,
                'error': np.inf,
                'dt': dt,
                'X': None,
                't': None
            }
        
        # Concatenate
        if X_all is None:
            X_all = X_chunk
            t_all = t_chunk
        else:
            X_all = np.hstack([X_all, X_chunk[:, 1:]])
            t_all = np.concatenate([t_all, t_chunk[1:]])
        
        x0 = X_chunk[:, -1]
        total_periods += chunk_periods
        
        # Check convergence
        converged, conv_period, max_diff = check_periodic_convergence(
            X_all, t_all, period, total_periods / period, tol=1e-3
        )
        
        x_periodic_est = X_all[:, -1]
        error = compute_periodic_error(x_periodic_est, x_periodic_ref)
        
        # Check if we've reached error target (if specified)
        if error_target is not None and error < error_target:
            comp_time = time.time() - start_time
            return {
                'method': 'Trapezoidal',
                'converged': True,
                'time': comp_time,
                'periods': total_periods,
                'error': error,
                'dt': dt,
                'X': X_all,
                't': t_all
            }
        
        if converged:
            comp_time = time.time() - start_time
            return {
                'method': 'Trapezoidal',
                'converged': True,
                'time': comp_time,
                'periods': total_periods,
                'error': error,
                'dt': dt,
                'X': X_all,
                't': t_all
            }
    
    comp_time = time.time() - start_time
    # Compute final error if not already computed
    if X_all is not None:
        x_periodic_est = X_all[:, -1]
        error = compute_periodic_error(x_periodic_est, x_periodic_ref)
    else:
        error = np.inf
    
    return {
        'method': 'Trapezoidal',
        'converged': False,
        'time': comp_time,
        'periods': total_periods,
        'error': error,
        'dt': dt,
        'X': X_all,
        't': t_all
    }


def shooting_newton_at_error_target(evalf, eval_Jf, x0_guess, p, eval_u, period,
                                     dt, x_periodic_ref, error_target):
    """Run Shooting-Newton with given dt, measure error."""
    start_time = time.time()
    
    x_periodic, info = periodic_steady_state_solver(
        evalf, eval_Jf, x0_guess, p, eval_u,
        period, dt, tol=1e-8, max_iter=50, verbose=False
    )
    
    comp_time = time.time() - start_time
    
    if info['converged']:
        error = compute_periodic_error(x_periodic, x_periodic_ref)
    else:
        error = np.inf
    
    return {
        'method': 'Shooting-Newton',
        'converged': info['converged'],
        'time': comp_time,
        'error': error,
        'dt': dt,
        'x_periodic': x_periodic,
        'newton_iters': info.get('iters', 0)
    }


def compare_methods():
    """Run comprehensive comparison."""
    print("="*70)
    print("PERIODIC STEADY-STATE METHOD COMPARISON")
    print("="*70)
    
    # Setup
    p, x0, _ = setup_12_genotype_model()
    period = 1.0
    pulse_duration = 0.1
    eval_u_periodic = create_periodic_input(
        uR_base=0.5, uC_pulse=2.0, pulse_duration=pulse_duration, period=period
    )
    
    # Step 1: Compute reference solution
    print("\n[1/5] Computing reference periodic steady state...")
    dt_ref = 0.001  # Very small dt for reference
    x_periodic_ref, dt_ref_actual, error_ref = find_periodic_ref_solution(
        evalf, evaljacobianf, x0, p, eval_u_periodic, period, dt_ref
    )
    print(f"  Reference error (periodic condition): {error_ref:.6e}")
    
    # Step 2: Test different error targets
    error_targets = [1e-2, 1e-3, 1e-4, 1e-5]
    
    print(f"\n[2/5] Testing Shooting-Newton at different dt values...")
    
    # Test Shooting-Newton with different dt values
    dt_values_sn = [0.01, 0.005, 0.002, 0.001]
    shooting_results = []
    
    # Initial guess for Shooting-Newton
    X_guess, _ = forward_euler(evalf, x0, p, eval_u_periodic, 0.0, 20*period, 0.01, verbose=False)
    x0_sn_guess = X_guess[:, -1]
    
    for dt_sn in dt_values_sn:
        print(f"  Testing dt = {dt_sn:.4f}...")
        result = shooting_newton_at_error_target(
            evalf, evaljacobianf, x0_sn_guess, p, eval_u_periodic, period,
            dt_sn, x_periodic_ref, None
        )
        shooting_results.append(result)
        print(f"    Error: {result['error']:.6e}, Time: {result['time']:.4f} s")
    
    # Step 3: Test Forward Euler transient
    print(f"\n[3/5] Testing Forward Euler transient at different dt (max 200 periods)...")
    
    dt_values_fe = [0.01, 0.005, 0.002]
    fe_results = []
    
    for dt_fe in dt_values_fe:
        print(f"  Testing dt = {dt_fe:.4f}...")
        # Run for full 200 periods and check error at end
        result = transient_forward_euler_to_periodic(
            evalf, x0, p, eval_u_periodic, period, dt_fe,
            x_periodic_ref, None, max_periods=200, verbose=True
        )
        fe_results.append(result)
        print(f"    Final error: {result['error']:.6e}, "
              f"Time: {result['time']:.4f} s, Periods: {result['periods']:.1f}")
        print(f"    Converged: {result['converged']}")
    
    # Step 4: Test Trapezoidal transient
    print(f"\n[4/5] Testing Trapezoidal transient at different dt (max 200 periods)...")
    
    dt_values_trap = [0.05, 0.02, 0.01]
    trap_results = []
    
    for dt_trap in dt_values_trap:
        print(f"  Testing dt = {dt_trap:.4f}...")
        # Run for full 200 periods and check error at end
        result = transient_trapezoidal_to_periodic(
            evalf, evaljacobianf, x0, p, eval_u_periodic, period, dt_trap,
            x_periodic_ref, None, max_periods=200, verbose=True
        )
        if result['X'] is not None:  # Only add if successful
            trap_results.append(result)
            print(f"    Final error: {result['error']:.6e}, "
                  f"Time: {result['time']:.4f} s, Periods: {result['periods']:.1f}")
            print(f"    Converged: {result['converged']}")
        else:
            print(f"    Failed (Newton convergence issues)")
    
    # Step 5: Analysis and plotting
    print(f"\n[5/5] Analyzing results...")
    
    # Organize results by error level
    comparison_data = {
        'reference': {
            'error': error_ref,
            'dt': dt_ref_actual
        },
        'shooting_newton': shooting_results,
        'forward_euler': fe_results,
        'trapezoidal': trap_results
    }
    
    # Save results
    with open('PM5/periodic_method_comparison_results.json', 'w') as f:
        # Convert to JSON-serializable
        json_data = {
            'reference': {
                'error': float(error_ref),
                'dt': float(dt_ref_actual)
            },
            'shooting_newton': [
                {
                    'error': float(r['error']),
                    'time': float(r['time']),
                    'dt': float(r['dt']),
                    'converged': bool(r['converged']),
                    'newton_iters': int(r.get('newton_iters', 0))
                }
                for r in shooting_results
            ],
            'forward_euler': [
                {
                    'error': float(r['error']),
                    'time': float(r['time']),
                    'dt': float(r['dt']),
                    'periods': float(r['periods']),
                    'converged': bool(r['converged'])
                }
                for r in fe_results
            ],
            'trapezoidal': [
                {
                    'error': float(r['error']),
                    'time': float(r['time']),
                    'dt': float(r['dt']),
                    'periods': float(r['periods']),
                    'converged': bool(r['converged'])
                }
                for r in trap_results
            ]
        }
        json.dump(json_data, f, indent=2)
    
    print(f"\nResults saved to: PM5/periodic_method_comparison_results.json")
    
    return comparison_data


def plot_comparison(comparison_data, save_path=None):
    """Plot comparison of methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sn_results = comparison_data['shooting_newton']
    fe_results = comparison_data['forward_euler']
    trap_results = comparison_data['trapezoidal']
    
    # Extract data
    sn_errors = [r['error'] for r in sn_results if r['converged']]
    sn_times = [r['time'] for r in sn_results if r['converged']]
    
    fe_errors = [r['error'] for r in fe_results]
    fe_times = [r['time'] for r in fe_results]
    
    trap_errors = [r['error'] for r in trap_results]
    trap_times = [r['time'] for r in trap_results]
    
    # Plot 1: Error vs Computation Time
    ax = axes[0, 0]
    if sn_errors:
        ax.loglog(sn_errors, sn_times, 'o-', linewidth=2, markersize=8, 
                 label='Shooting-Newton', color='blue', alpha=0.8)
    if fe_errors:
        ax.loglog(fe_errors, fe_times, 's-', linewidth=2, markersize=8,
                 label='Forward Euler Transient', color='green', alpha=0.8)
    if trap_errors:
        ax.loglog(trap_errors, trap_times, '^-', linewidth=2, markersize=8,
                 label='Trapezoidal Transient', color='orange', alpha=0.8)
    
    ax.set_xlabel('Error $||x - x_{\\text{ref}}||_\\infty$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Error vs Computation Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Computation Time Comparison (at similar error levels)
    ax = axes[0, 1]
    
    # Find comparable error levels
    all_errors = sorted(set(sn_errors + fe_errors + trap_errors))
    
    # Interpolate times at common error levels
    error_levels = [1e-2, 1e-3, 1e-4, 1e-5]
    methods = ['Shooting-Newton', 'Forward Euler', 'Trapezoidal']
    times_at_errors = {method: [] for method in methods}
    
    for err_target in error_levels:
        # Find closest results
        sn_time = None
        for r in sn_results:
            if r['converged'] and abs(np.log10(r['error']) - np.log10(err_target)) < 0.5:
                sn_time = r['time']
                break
        
        fe_time = None
        for r in fe_results:
            if abs(np.log10(r['error']) - np.log10(err_target)) < 0.5:
                fe_time = r['time']
                break
        
        trap_time = None
        for r in trap_results:
            if abs(np.log10(r['error']) - np.log10(err_target)) < 0.5:
                trap_time = r['time']
                break
        
        if sn_time:
            times_at_errors['Shooting-Newton'].append(sn_time)
        else:
            times_at_errors['Shooting-Newton'].append(None)
        
        if fe_time:
            times_at_errors['Forward Euler'].append(fe_time)
        else:
            times_at_errors['Forward Euler'].append(None)
        
        if trap_time:
            times_at_errors['Trapezoidal'].append(trap_time)
        else:
            times_at_errors['Trapezoidal'].append(None)
    
    x = np.arange(len(error_levels))
    width = 0.25
    
    colors = {'Shooting-Newton': 'blue', 'Forward Euler': 'green', 'Trapezoidal': 'orange'}
    
    for i, method in enumerate(methods):
        times = times_at_errors[method]
        valid_times = [t if t is not None else 0 for t in times]
        valid_mask = [t is not None for t in times]
        
        if any(valid_mask):
            bars = ax.bar(x + i*width, valid_times, width, label=method, 
                         color=colors[method], alpha=0.7, edgecolor='black')
            # Mark missing data
            for j, (bar, valid) in enumerate(zip(bars, valid_mask)):
                if not valid:
                    bar.set_hatch('///')
                    bar.set_alpha(0.3)
    
    ax.set_xlabel('Error Target', fontsize=12, fontweight='bold')
    ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Computation Time at Different Error Levels', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'$10^{{{int(np.log10(e))}}}$' for e in error_levels])
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Speedup factor (relative to Forward Euler)
    ax = axes[1, 0]
    
    # Compute speedups at common error levels
    speedups = {method: [] for method in methods}
    
    for err_target in error_levels:
        # Find Forward Euler time
        fe_time = None
        for r in fe_results:
            if abs(np.log10(r['error']) - np.log10(err_target)) < 0.5:
                fe_time = r['time']
                break
        
        if fe_time:
            # Shooting-Newton speedup
            sn_time = None
            for r in sn_results:
                if r['converged'] and abs(np.log10(r['error']) - np.log10(err_target)) < 0.5:
                    sn_time = r['time']
                    break
            speedups['Shooting-Newton'].append(fe_time / sn_time if sn_time else None)
            
            # Trapezoidal speedup
            trap_time = None
            for r in trap_results:
                if abs(np.log10(r['error']) - np.log10(err_target)) < 0.5:
                    trap_time = r['time']
                    break
            speedups['Trapezoidal'].append(fe_time / trap_time if trap_time else None)
        else:
            speedups['Shooting-Newton'].append(None)
            speedups['Trapezoidal'].append(None)
    
    x = np.arange(len(error_levels))
    width = 0.35
    
    for i, method in enumerate(['Shooting-Newton', 'Trapezoidal']):
        sp = speedups[method]
        valid_sp = [s if s is not None else 0 for s in sp]
        valid_mask = [s is not None for s in sp]
        
        if any(valid_mask):
            bars = ax.bar(x + i*width, valid_sp, width, label=method,
                         color=colors[method], alpha=0.7, edgecolor='black')
            # Mark missing
            for j, (bar, valid) in enumerate(zip(bars, valid_mask)):
                if not valid:
                    bar.set_hatch('///')
                    bar.set_alpha(0.3)
    
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Forward Euler (baseline)')
    ax.set_xlabel('Error Target', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Relative to Forward Euler Transient', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([f'$10^{{{int(np.log10(e))}}}$' for e in error_levels])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = "Summary Statistics\n" + "="*50 + "\n\n"
    summary_text += f"Reference Solution:\n"
    summary_text += f"  Error: {comparison_data['reference']['error']:.6e}\n"
    summary_text += f"  dt: {comparison_data['reference']['dt']:.6e}\n\n"
    
    summary_text += f"Shooting-Newton:\n"
    if sn_results:
        min_time = min([r['time'] for r in sn_results if r['converged']])
        min_error = min([r['error'] for r in sn_results if r['converged']])
        summary_text += f"  Min time: {min_time:.4f} s\n"
        summary_text += f"  Min error: {min_error:.6e}\n"
    
    summary_text += f"\nForward Euler Transient:\n"
    if fe_results:
        min_time = min([r['time'] for r in fe_results])
        min_error = min([r['error'] for r in fe_results])
        summary_text += f"  Min time: {min_time:.4f} s\n"
        summary_text += f"  Min error: {min_error:.6e}\n"
    
    summary_text += f"\nTrapezoidal Transient:\n"
    if trap_results:
        min_time = min([r['time'] for r in trap_results])
        min_error = min([r['error'] for r in trap_results])
        summary_text += f"  Min time: {min_time:.4f} s\n"
        summary_text += f"  Min error: {min_error:.6e}\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    
    plt.suptitle('Periodic Steady-State Method Comparison', 
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == '__main__':
    comparison_data = compare_methods()
    plot_comparison(comparison_data, save_path='PM5/plot11_periodic_method_comparison.png')
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)

