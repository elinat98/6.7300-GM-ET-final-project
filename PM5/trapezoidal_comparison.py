#!/usr/bin/env python3
"""
Comprehensive Trapezoidal method comparisons:
1. Fixed vs Adaptive time-stepping
2. Initialization methods: Previous time step vs Forward Euler prediction
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
from PM5.adaptive_trapezoidal import adaptive_trapezoidal


def compare_initialization_methods(eval_f, eval_Jf, x0, p, eval_u, t_start, t_stop, dt,
                                   verbose=True):
    """
    Compare Trapezoidal with different initialization methods.
    
    Methods:
    1. Previous time step: x_guess = x_n
    2. Forward Euler prediction: x_guess = x_n + dt * f(x_n)
    """
    
    print("="*70)
    print("COMPARING TRAPEZOIDAL INITIALIZATION METHODS")
    print("="*70)
    print(f"Time step: dt = {dt:.6e}")
    print(f"Time range: [{t_start}, {t_stop}]")
    
    results = {}
    
    for init_method in ['previous', 'feuler']:
        print(f"\n[{'1' if init_method == 'previous' else '2'}/2] Testing: {init_method}")
        print("-" * 70)
        
        start_time = time.time()
        newton_iterations = []
        convergence_failures = 0
        
        try:
            X, t, newton_stats = trapezoidal_fallback(
                eval_f, eval_Jf, x0, p, eval_u,
                t_start, t_stop, dt,
                init_method=init_method, verbose=verbose
            )
            
            comp_time = time.time() - start_time
            
            if X is None:
                print(f"  ✗ Failed")
                results[init_method] = {
                    'success': False,
                    'time': None,
                    'error': None,
                    'newton_iters': None
                }
                continue
            
            num_steps = len(t) - 1
            avg_newton_iters = newton_stats.get('avg_iterations', 0) if newton_stats else 0
            max_newton_iters = newton_stats.get('max_iterations', 0) if newton_stats else 0
            
            results[init_method] = {
                'success': True,
                'time': comp_time,
                'steps': num_steps,
                'X': X,
                't': t,
                'avg_newton_iters': avg_newton_iters,
                'max_newton_iters': max_newton_iters,
                'newton_stats': newton_stats
            }
            
            print(f"  ✓ Success")
            print(f"  Computation time: {comp_time:.4f} seconds")
            print(f"  Number of steps: {num_steps}")
            print(f"  Avg Newton iterations: {avg_newton_iters:.2f}")
            print(f"  Max Newton iterations: {max_newton_iters}")
            if newton_stats:
                print(f"  Convergence failures: {newton_stats.get('failures', 0)}")
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results[init_method] = {
                'success': False,
                'error': str(e)
            }
    
    # Compare results
    if results.get('previous', {}).get('success') and results.get('feuler', {}).get('success'):
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        time_prev = results['previous']['time']
        time_fe = results['feuler']['time']
        speedup = time_prev / time_fe if time_fe > 0 else np.inf
        
        print(f"Previous time step initialization:")
        print(f"  Time: {time_prev:.4f} s")
        print(f"\nForward Euler prediction initialization:")
        print(f"  Time: {time_fe:.4f} s")
        print(f"\nSpeedup: {speedup:.2f}x ({'FE faster' if speedup > 1 else 'Previous faster'})")
    
    return results


def compare_fixed_vs_adaptive_trapezoidal(eval_f, eval_Jf, x0, p, eval_u, t_start, t_stop,
                                          xref, error_target, dt_fixed=None, verbose=True):
    """
    Compare fixed vs adaptive time-stepping for Trapezoidal.
    """
    
    print("="*70)
    print("COMPARING FIXED vs ADAPTIVE TRAPEZOIDAL")
    print("="*70)
    print(f"Time range: [{t_start}, {t_stop}]")
    print(f"Error target: ϵa = {error_target:.6e}")
    
    # Find fixed dt that gives approximately error_target
    if dt_fixed is None:
        print("\n[1/4] Finding fixed dt for error target...")
        
        # Binary search for fixed dt
        dt_low = 1e-3
        dt_high = 0.5
        dt_fixed = 0.1  # Initial guess
        
        best_dt = None
        best_error = np.inf
        
        for i in range(15):
            dt_test = np.sqrt(dt_low * dt_high)
            
            try:
                X_test, t_test, _ = trapezoidal_fallback(
                    eval_f, eval_Jf, x0, p, eval_u,
                    t_start, t_stop, dt_test,
                    init_method='feuler', verbose=False
                )
                
                if X_test is None:
                    dt_high = dt_test
                    continue
                
                error_test = np.linalg.norm(X_test[:, -1] - xref, ord=np.inf)
                
                if abs(error_test - error_target) < abs(best_error - error_target):
                    best_dt = dt_test
                    best_error = error_test
                
                if abs(error_test - error_target) / error_target < 0.3:  # Within 30%
                    dt_fixed = dt_test
                    break
                
                if error_test > error_target:
                    dt_high = dt_test
                else:
                    dt_low = dt_test
                    
            except Exception:
                dt_high = dt_test
                continue
        
        if dt_fixed == 0.1 and best_dt is not None:
            dt_fixed = best_dt
        
        print(f"  Fixed dt: {dt_fixed:.6e}")
    
    # Run fixed Trapezoidal
    print("\n[2/4] Running fixed Trapezoidal...")
    start_time = time.time()
    try:
        X_fixed, t_fixed, newton_stats_fixed = trapezoidal_fallback(
            eval_f, eval_Jf, x0, p, eval_u,
            t_start, t_stop, dt_fixed,
            init_method='feuler', verbose=False
        )
        time_fixed = time.time() - start_time
        
        if X_fixed is None:
            print("  ✗ Fixed Trapezoidal failed")
            return None
        
        error_fixed = np.linalg.norm(X_fixed[:, -1] - xref, ord=np.inf)
        steps_fixed = len(t_fixed) - 1
        
        print(f"  ✓ Success")
        print(f"  Time: {time_fixed:.4f} seconds")
        print(f"  Steps: {steps_fixed}")
        print(f"  Error: {error_fixed:.6e}")
        
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return None
    
    # Run adaptive Trapezoidal
    print("\n[3/4] Running adaptive Trapezoidal...")
    start_time = time.time()
    try:
        X_adaptive, t_adaptive, stats_adaptive = adaptive_trapezoidal(
            eval_f, eval_Jf, x0, p, eval_u,
            t_start, t_stop,
            dt_initial=dt_fixed,
            dt_min=1e-4,
            dt_max=0.5,
            error_target=error_target,
            xref=xref,
            verbose=verbose
        )
        time_adaptive = time.time() - start_time
        
        error_adaptive = np.linalg.norm(X_adaptive[:, -1] - xref, ord=np.inf)
        steps_adaptive = stats_adaptive['steps_accepted']
        
        print(f"  ✓ Success")
        print(f"  Time: {time_adaptive:.4f} seconds")
        print(f"  Steps accepted: {steps_adaptive}")
        print(f"  Steps rejected: {stats_adaptive['steps_rejected']}")
        print(f"  Error: {error_adaptive:.6e}")
        print(f"  Avg dt: {np.mean(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0:.6e}")
        print(f"  Min dt: {np.min(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0:.6e}")
        print(f"  Max dt: {np.max(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0:.6e}")
        
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Comparison
    speedup = time_fixed / time_adaptive if time_adaptive > 0 else np.inf
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"Fixed Trapezoidal:")
    print(f"  Time: {time_fixed:.4f} s")
    print(f"  Steps: {steps_fixed}")
    print(f"  Error: {error_fixed:.6e}")
    print(f"  dt: {dt_fixed:.6e}")
    print(f"\nAdaptive Trapezoidal:")
    print(f"  Time: {time_adaptive:.4f} s")
    print(f"  Steps: {steps_adaptive}")
    print(f"  Rejections: {stats_adaptive['steps_rejected']}")
    print(f"  Error: {error_adaptive:.6e}")
    print(f"  Avg dt: {np.mean(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0:.6e}")
    print(f"\nSpeedup: {speedup:.2f}x ({'adaptive faster' if speedup > 1 else 'adaptive slower'})")
    
    return {
        'fixed': {
            'dt': dt_fixed,
            'time': time_fixed,
            'steps': steps_fixed,
            'error': error_fixed,
            'X': X_fixed,
            't': t_fixed
        },
        'adaptive': {
            'time': time_adaptive,
            'steps': steps_adaptive,
            'rejections': stats_adaptive['steps_rejected'],
            'error': error_adaptive,
            'dt_avg': np.mean(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0,
            'dt_min': np.min(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0,
            'dt_max': np.max(stats_adaptive['dt_values']) if stats_adaptive['dt_values'] else 0,
            'stats': stats_adaptive,
            'X': X_adaptive,
            't': t_adaptive
        },
        'speedup': speedup
    }


def plot_initialization_comparison(init_results, save_path=None):
    """Plot comparison of initialization methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if not (init_results.get('previous', {}).get('success') and 
            init_results.get('feuler', {}).get('success')):
        axes[0, 0].text(0.5, 0.5, 'Comparison data not available', 
                       transform=axes[0, 0].transAxes, ha='center')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    # Plot 1: Computation time comparison
    methods = ['Previous\nTime Step', 'Forward Euler\nPrediction']
    times = [init_results['previous']['time'], init_results['feuler']['time']]
    colors = ['blue', 'orange']
    
    axes[0, 0].bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Computation Time (seconds)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Computation Time Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    for i, (method, t) in enumerate(zip(methods, times)):
        axes[0, 0].text(i, t, f'{t:.4f} s', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    # Plot 2: Solution trajectories
    X_prev = init_results['previous']['X']
    t_prev = init_results['previous']['t']
    X_fe = init_results['feuler']['X']
    t_fe = init_results['feuler']['t']
    
    m = 12
    total_biomass_prev = np.sum(X_prev[:m, :], axis=0)
    total_biomass_fe = np.sum(X_fe[:m, :], axis=0)
    
    axes[0, 1].plot(t_prev, total_biomass_prev, '-', linewidth=2, 
                   label='Previous time step', color='blue', alpha=0.8)
    axes[0, 1].plot(t_fe, total_biomass_fe, '--', linewidth=2,
                   label='Forward Euler prediction', color='orange', alpha=0.8)
    axes[0, 1].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Total Biomass', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Solution Trajectories', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Difference between methods
    # Interpolate to common time points
    t_common = np.linspace(t_prev[0], t_prev[-1], 100)
    X_prev_interp = np.array([np.interp(t_common, t_prev, X_prev[i, :]) 
                             for i in range(X_prev.shape[0])])
    X_fe_interp = np.array([np.interp(t_common, t_fe, X_fe[i, :]) 
                            for i in range(X_fe.shape[0])])
    
    diff = np.linalg.norm(X_prev_interp - X_fe_interp, axis=0, ord=np.inf)
    axes[1, 0].plot(t_common, diff, '-', linewidth=2, color='red', alpha=0.8)
    axes[1, 0].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Difference $||x_{prev} - x_{FE}||_\\infty$', 
                         fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Solution Difference', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Speedup factor
    speedup = init_results['previous']['time'] / init_results['feuler']['time']
    axes[1, 1].bar(['Speedup Factor'], [speedup], color='green', alpha=0.7,
                  edgecolor='black', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No speedup')
    axes[1, 1].set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Forward Euler Prediction Speedup', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].text(0, speedup, f'{speedup:.2f}x', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_adaptive_comparison(adaptive_results, save_path=None):
    """Plot comparison of fixed vs adaptive Trapezoidal."""
    if adaptive_results is None:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    fixed = adaptive_results['fixed']
    adaptive = adaptive_results['adaptive']
    
    # Plot 1: Computation time
    methods = ['Fixed\nTrapezoidal', 'Adaptive\nTrapezoidal']
    times = [fixed['time'], adaptive['time']]
    colors = ['blue', 'orange']
    
    axes[0, 0].bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Computation Time (seconds)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Computation Time', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, (method, t) in enumerate(zip(methods, times)):
        axes[0, 0].text(i, t, f'{t:.4f} s', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    # Plot 2: Number of steps
    steps = [fixed['steps'], adaptive['steps']]
    axes[0, 1].bar(methods, steps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Number of Steps', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Number of Steps', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, (method, s) in enumerate(zip(methods, steps)):
        axes[0, 1].text(i, s, f'{s}', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    # Plot 3: Error comparison
    errors = [fixed['error'], adaptive['error']]
    axes[0, 2].bar(methods, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 2].set_ylabel('Error $||x - x_{ref}||_\\infty$', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('Error Comparison', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    for i, (method, e) in enumerate(zip(methods, errors)):
        axes[0, 2].text(i, e, f'{e:.6e}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    
    # Plot 4: Time step evolution (adaptive)
    if adaptive['stats']['dt_values']:
        dt_values = adaptive['stats']['dt_values']
        step_indices = np.arange(len(dt_values))
        axes[1, 0].plot(step_indices, dt_values, '-', linewidth=2, color='orange', alpha=0.8)
        axes[1, 0].axhline(y=fixed['dt'], color='blue', linestyle='--', linewidth=2,
                          label=f'Fixed dt = {fixed["dt"]:.6e}')
        axes[1, 0].set_xlabel('Step Index', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Time Step $\\Delta t$', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Adaptive Time Step Evolution', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Solution trajectories
    m = 12
    total_biomass_fixed = np.sum(fixed['X'][:m, :], axis=0)
    total_biomass_adaptive = np.sum(adaptive['X'][:m, :], axis=0)
    
    axes[1, 1].plot(fixed['t'], total_biomass_fixed, '-', linewidth=2,
                   label='Fixed Trapezoidal', color='blue', alpha=0.8)
    axes[1, 1].plot(adaptive['t'], total_biomass_adaptive, '--', linewidth=2,
                   label='Adaptive Trapezoidal', color='orange', alpha=0.8)
    axes[1, 1].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Total Biomass', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Solution Trajectories', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Speedup
    speedup = adaptive_results['speedup']
    axes[1, 2].bar(['Speedup'], [speedup], color='green' if speedup > 1 else 'red',
                  alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 2].axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    axes[1, 2].set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    axes[1, 2].set_title('Adaptive vs Fixed Speedup', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].text(0, speedup, f'{speedup:.2f}x', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def main():
    """Run all Trapezoidal comparisons."""
    print("="*70)
    print("TRAPEZOIDAL METHOD COMPREHENSIVE COMPARISON")
    print("="*70)
    
    # Set up model
    print("\n[1/3] Setting up model...")
    p, x0, eval_u = setup_12_genotype_model()
    t_start, t_stop = 0.0, 5.0
    
    # Load reference solution
    print("\n[2/3] Loading reference solution...")
    with open('PM5/stability_results.json', 'r') as f:
        ref_data = json.load(f)
    xref = np.array(ref_data['xref'])
    error_target = ref_data['eps_a']
    
    # Comparison 1: Initialization methods
    print("\n" + "="*70)
    print("COMPARISON 1: INITIALIZATION METHODS")
    print("="*70)
    # Use smaller dt for better convergence
    dt_test = 0.05
    init_results = compare_initialization_methods(
        evalf, evaljacobianf, x0, p, eval_u,
        t_start, t_stop, dt_test, verbose=True
    )
    
    # Plot initialization comparison
    plot_initialization_comparison(init_results, 
                                  save_path='PM5/plot7_initialization_comparison.png')
    
    # Save results
    with open('PM5/initialization_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON
        def convert_to_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json(item) for item in obj]
            else:
                return obj
        
        json_data = convert_to_json(init_results)
        json.dump(json_data, f, indent=2)
    
    # Comparison 2: Fixed vs Adaptive
    print("\n" + "="*70)
    print("COMPARISON 2: FIXED vs ADAPTIVE TIME-STEPPING")
    print("="*70)
    adaptive_results = compare_fixed_vs_adaptive_trapezoidal(
        evalf, evaljacobianf, x0, p, eval_u,
        t_start, t_stop, xref, error_target,
        dt_fixed=None, verbose=True
    )
    
    # Plot adaptive comparison
    if adaptive_results:
        plot_adaptive_comparison(adaptive_results,
                                save_path='PM5/plot8_adaptive_comparison.png')
        
        # Save results
        def convert_to_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json(item) for item in obj]
            else:
                return obj
        
        with open('PM5/adaptive_comparison_results.json', 'w') as f:
            json_data = convert_to_json(adaptive_results)
            json.dump(json_data, f, indent=2)
    
    print("\n" + "="*70)
    print("ALL COMPARISONS COMPLETE")
    print("="*70)
    print("\nPlots saved:")
    print("  - plot7_initialization_comparison.png")
    if adaptive_results:
        print("  - plot8_adaptive_comparison.png")


if __name__ == '__main__':
    main()

