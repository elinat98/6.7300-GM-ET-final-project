#!/usr/bin/env python3
"""
Analyze periodic steady-state behavior with periodic antibiotic boluses.

This script:
1. Sets up periodic input (periodic antibiotic boluses)
2. Runs Forward Euler simulation for many periods
3. Monitors convergence to periodic steady state
4. Compares time-transient simulation vs periodic steady-state solver
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
from PM5.model_setup import setup_12_genotype_model
from PM5.forward_euler import forward_euler
from PM4.newton.NewtonSolver import newton_solve


def create_periodic_input(uR_base=0.5, uC_pulse=2.0, pulse_duration=0.1, period=1.0):
    """
    Create periodic input function with antibiotic boluses.
    
    Parameters
    ----------
    uR_base : float
        Constant resource supply rate
    uC_pulse : float
        Peak antibiotic concentration during pulse
    pulse_duration : float
        Duration of each pulse (fraction of period)
    period : float
        Period of antibiotic administration
    
    Returns
    -------
    eval_u : callable
        Input function u(t) = [uR, uC]
    """
    def eval_u(t):
        # Resource is constant
        uR = uR_base
        
        # Antibiotic is periodic pulses
        t_mod = t % period
        if t_mod < pulse_duration:
            # Pulse phase
            uC = uC_pulse
        else:
            # No antibiotic
            uC = 0.0
        
        return np.array([uR, uC])
    
    return eval_u


def check_periodic_convergence(X, t, period, num_periods, tol=1e-3):
    """
    Check if solution has converged to periodic steady state.
    
    Compares state at corresponding times in consecutive periods.
    
    Parameters
    ----------
    X : array, shape (N, num_steps)
        State trajectory
    t : array, shape (num_steps,)
        Time points
    period : float
        Period of input
    num_periods : float
        Number of periods simulated
    tol : float
        Convergence tolerance
    
    Returns
    -------
    converged : bool
        Whether solution has converged
    convergence_period : float
        Period at which convergence occurs (if converged)
    max_diff : array
        Maximum difference between periods at each comparison point
    """
    if num_periods < 2:
        return False, None, None
    
    # Find indices corresponding to same phase in different periods
    # Compare period (n-1) to period n
    period_n_minus_1 = period * (num_periods - 2)
    period_n = period * (num_periods - 1)
    
    # Sample points within one period to compare
    num_samples = min(20, len(t))
    sample_times = np.linspace(period_n_minus_1, period_n_minus_1 + period, num_samples)
    
    max_diffs = []
    
    for t_sample in sample_times:
        if t_sample > t[-1] or t_sample + period > t[-1]:
            continue
        
        # Interpolate state at these times
        idx_prev = np.searchsorted(t, t_sample, side='right') - 1
        idx_next = min(idx_prev + 1, len(t) - 1)
        
        if idx_prev < 0:
            continue
        
        # Linear interpolation
        if idx_prev == idx_next:
            x_prev = X[:, idx_prev]
        else:
            alpha = (t_sample - t[idx_prev]) / (t[idx_next] - t[idx_prev] + 1e-10)
            x_prev = (1 - alpha) * X[:, idx_prev] + alpha * X[:, idx_next]
        
        # Same phase in next period
        t_next = t_sample + period
        idx_prev_next = np.searchsorted(t, t_next, side='right') - 1
        idx_next_next = min(idx_prev_next + 1, len(t) - 1)
        
        if idx_prev_next < 0:
            continue
        
        if idx_prev_next == idx_next_next:
            x_next = X[:, idx_prev_next]
        else:
            alpha = (t_next - t[idx_prev_next]) / (t[idx_next_next] - t[idx_prev_next] + 1e-10)
            x_next = (1 - alpha) * X[:, idx_prev_next] + alpha * X[:, idx_next_next]
        
        # Compute difference
        diff = np.linalg.norm(x_prev - x_next, ord=np.inf)
        max_diffs.append(diff)
    
    if len(max_diffs) == 0:
        return False, None, None
    
    max_diff = max(max_diffs)
    converged = max_diff < tol
    
    if converged:
        convergence_period = num_periods - 1  # Converged between last two periods
    else:
        convergence_period = None
    
    return converged, convergence_period, max_diff


def state_transition(eval_f, x0, p, eval_u, t_start, t_stop, dt):
    """
    State transition function: Φ(x0, T) = x(T)
    
    Integrates ODE from t_start to t_stop with initial condition x0.
    
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
    dt : float
        Time step
    
    Returns
    -------
    x_final : array
        State at t_stop
    """
    from PM5.forward_euler import forward_euler
    X, t = forward_euler(eval_f, x0, p, eval_u, t_start, t_stop, dt, verbose=False)
    return X[:, -1]


def periodic_steady_state_solver(eval_f, eval_Jf, x0_guess, p, eval_u, period, dt_ref, 
                                  tol=1e-6, max_iter=50, verbose=False):
    """
    Solve for periodic steady state using Shooting-Newton method.
    
    Find x0 such that Φ(x0, period) = x0, i.e., x(period) = x(0) for periodic input u(t).
    
    Uses Newton's method to solve:
        R(x0) = Φ(x0, period) - x0 = 0
    
    Parameters
    ----------
    eval_f : callable
        ODE right-hand side
    eval_Jf : callable
        Jacobian of f (for sensitivity analysis, not used here)
    x0_guess : array
        Initial guess for periodic steady state
    p : dict
        Model parameters
    eval_u : callable
        Periodic input function
    period : float
        Period of input (T)
    dt_ref : float
        Time step for integration
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum Newton iterations
    verbose : bool
        If True, print progress
    
    Returns
    -------
    x_periodic : array
        Periodic steady state initial condition (x(0) = x(period))
    info : dict
        Convergence information
    """
    N = len(x0_guess)
    
    # Define residual: R(x0) = Φ(x0, period) - x0 = 0
    def residual(x0, p_dummy, u_dummy):
        x0_flat = np.asarray(x0, dtype=float).ravel()
        # State transition: integrate over one period
        x_period_end = state_transition(eval_f, x0_flat, p, eval_u, 0.0, period, dt_ref)
        # Residual: x(period) - x(0) should be zero
        return (x_period_end - x0_flat).reshape((-1, 1))
    
    # Jacobian of residual: dR/dx0 = dΦ/dx0 - I
    # Use finite differences for sensitivity matrix
    def residual_jacobian(x0, p_dummy, u_dummy):
        x0_flat = np.asarray(x0, dtype=float).ravel()
        N = len(x0_flat)
        eps = 1e-6
        
        # Compute dΦ/dx0 using finite differences
        dPhi_dx0 = np.zeros((N, N))
        x_period_0 = state_transition(eval_f, x0_flat, p, eval_u, 0.0, period, dt_ref)
        
        for j in range(N):
            x0_perturbed = x0_flat.copy()
            x0_perturbed[j] += eps
            x_period_perturbed = state_transition(eval_f, x0_perturbed, p, eval_u, 0.0, period, dt_ref)
            dPhi_dx0[:, j] = (x_period_perturbed - x_period_0) / eps
        
        # dR/dx0 = dΦ/dx0 - I
        return dPhi_dx0 - np.eye(N)
    
    # Solve using Newton
    if verbose:
        print(f"  Shooting-Newton: finding x0 such that Φ(x0, {period}) = x0")
    
    x_periodic, info = newton_solve(
        residual, x0_guess.reshape((-1, 1)),
        {}, np.zeros(0),
        jacobian_fn=residual_jacobian,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )
    
    return x_periodic.ravel(), info


def run_periodic_analysis():
    """Run full periodic steady-state analysis."""
    print("="*70)
    print("PERIODIC STEADY-STATE ANALYSIS")
    print("="*70)
    
    # Setup model
    print("\n[1/5] Setting up model...")
    p, x0, _ = setup_12_genotype_model()
    
    # Create periodic input
    period = 1.0  # 1 time unit period
    pulse_duration = 0.1  # 10% of period is pulse
    uC_pulse = 2.0  # High antibiotic during pulse
    eval_u_periodic = create_periodic_input(
        uR_base=0.5,
        uC_pulse=uC_pulse,
        pulse_duration=pulse_duration,
        period=period
    )
    
    print(f"  Period: {period}")
    print(f"  Pulse duration: {pulse_duration} ({pulse_duration/period*100:.1f}% of period)")
    print(f"  Pulse amplitude: {uC_pulse}")
    
    # Forward Euler parameters
    dt = 0.01  # Small time step for accuracy
    num_periods = 50  # Simulate many periods
    t_stop = num_periods * period
    
    # Run transient simulation
    print(f"\n[2/5] Running transient simulation ({num_periods} periods)...")
    start_time = time.time()
    X_transient, t_transient = forward_euler(
        evalf, x0, p, eval_u_periodic,
        0.0, t_stop, dt, verbose=False
    )
    time_transient = time.time() - start_time
    
    print(f"  Simulation time: {time_transient:.4f} seconds")
    print(f"  Number of steps: {len(t_transient) - 1}")
    print(f"  Final time: {t_transient[-1]:.2f}")
    
    # Check convergence
    print(f"\n[3/5] Checking periodic convergence...")
    convergence_results = []
    
    for n_periods in range(2, num_periods + 1):
        converged, conv_period, max_diff = check_periodic_convergence(
            X_transient, t_transient, period, n_periods, tol=1e-3
        )
        convergence_results.append({
            'periods': float(n_periods),
            'time': float(n_periods * period),
            'converged': bool(converged),
            'max_diff': float(max_diff) if max_diff is not None else None
        })
        
        if converged and conv_period is not None:
            print(f"  ✓ Converged at period {conv_period:.1f}")
            print(f"    Max difference: {max_diff:.6e}")
            break
        
        if n_periods % 10 == 0:
            print(f"  Period {n_periods}: max_diff = {max_diff:.6e}")
    
    # Find convergence period
    convergence_period = None
    for result in convergence_results:
        if result['converged']:
            convergence_period = result['periods']
            break
    
    # Compare with periodic steady-state solver (Shooting-Newton)
    print(f"\n[4/5] Solving periodic steady state using Shooting-Newton...")
    
    # Use final state from transient as initial guess (better than initial state)
    x0_periodic = X_transient[:, -1].copy()
    
    from jacobian_tools import evaljacobianf
    
    start_time = time.time()
    try:
        x_periodic, info = periodic_steady_state_solver(
            evalf, evaljacobianf, x0_periodic, p, eval_u_periodic,
            period, dt, tol=1e-6, max_iter=30, verbose=True
        )
        time_periodic_solver = time.time() - start_time
        
        if info['converged']:
            print(f"\n  ✓ Converged in {info.get('iters', 0)} Newton iterations")
            print(f"  Solver time: {time_periodic_solver:.4f} seconds")
            
            # Verify: integrate one period from x_periodic
            X_verify, t_verify = forward_euler(
                evalf, x_periodic, p, eval_u_periodic,
                0.0, period, dt, verbose=False
            )
            error_verification = np.linalg.norm(X_verify[:, -1] - x_periodic, ord=np.inf)
            print(f"  Verification: ||x({period}) - x(0)||_∞ = {error_verification:.6e}")
            
            if error_verification < 1e-5:
                print(f"  ✓ Periodic condition satisfied!")
            else:
                print(f"  ⚠ Warning: Verification error is larger than expected")
        else:
            print(f"  ✗ Did not converge (residual norm: {info.get('residual_norm', 'unknown')})")
            time_periodic_solver = None
            x_periodic = None
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        time_periodic_solver = None
        x_periodic = None
    
    # Summary
    print(f"\n[5/5] Summary")
    print("="*70)
    
    if convergence_period:
        time_to_convergence = convergence_period * period
        print(f"Transient simulation:")
        print(f"  Periods to convergence: {convergence_period:.1f}")
        print(f"  Time to convergence: {time_to_convergence:.2f}")
        print(f"  Total simulation time: {time_transient:.4f} seconds")
        print(f"  Steps to convergence: {int(time_to_convergence / dt)}")
        print(f"  Effective time to convergence: {time_transient * (convergence_period / num_periods):.4f} seconds")
    else:
        print(f"Transient simulation:")
        print(f"  Did not converge within {num_periods} periods")
        print(f"  Total simulation time: {time_transient:.4f} seconds")
    
    if time_periodic_solver is not None:
        print(f"\nPeriodic steady-state solver:")
        print(f"  Solver time: {time_periodic_solver:.4f} seconds")
        
        if convergence_period:
            speedup = time_transient * (convergence_period / num_periods) / time_periodic_solver
            print(f"  Speedup: {speedup:.2f}x")
    
    # Save results (convert to JSON-serializable types)
    results = {
        'period': float(period),
        'pulse_duration': float(pulse_duration),
        'uC_pulse': float(uC_pulse),
        'dt': float(dt),
        'num_periods_simulated': int(num_periods),
        'time_transient': float(time_transient),
        'convergence_period': float(convergence_period) if convergence_period else None,
        'time_to_convergence': float(convergence_period * period) if convergence_period else None,
        'convergence_results': convergence_results,
        'time_periodic_solver': float(time_periodic_solver) if time_periodic_solver is not None else None,
        'periodic_solver_converged': bool(time_periodic_solver is not None),
        'x_periodic': x_periodic.tolist() if x_periodic is not None else None
    }
    
    with open('PM5/periodic_steady_state_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: PM5/periodic_steady_state_results.json")
    
    return results, X_transient, t_transient, x_periodic


def plot_periodic_convergence(X, t, period, num_periods, save_path=None):
    """Plot convergence to periodic steady state."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Total biomass over time
    m = 12
    total_biomass = np.sum(X[:m, :], axis=0)
    axes[0, 0].plot(t, total_biomass, '-', linewidth=1.5, alpha=0.7)
    axes[0, 0].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Total Biomass', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Total Biomass Evolution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark period boundaries
    for p_num in range(1, num_periods + 1):
        axes[0, 0].axvline(x=p_num * period, color='gray', linestyle='--', alpha=0.3)
    
    # Plot 2: Antibiotic concentration
    C = X[m+1, :]
    axes[0, 1].plot(t, C, '-', linewidth=1.5, alpha=0.7, color='red')
    axes[0, 1].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Antibiotic $C$', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Antibiotic Concentration', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Last few periods (zoomed)
    last_periods = 5
    t_start_zoom = (num_periods - last_periods) * period
    mask = t >= t_start_zoom
    
    axes[1, 0].plot(t[mask], total_biomass[mask], '-', linewidth=1.5, alpha=0.8)
    axes[1, 0].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Total Biomass', fontsize=11, fontweight='bold')
    axes[1, 0].set_title(f'Last {last_periods} Periods (Zoom)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Compare consecutive periods (if converged)
    if num_periods >= 2:
        period_n_minus_1 = (num_periods - 2) * period
        period_n = (num_periods - 1) * period
        
        # Extract one period from each
        mask1 = (t >= period_n_minus_1) & (t < period_n_minus_1 + period)
        mask2 = (t >= period_n) & (t < period_n + period)
        
        if np.any(mask1) and np.any(mask2):
            t1 = t[mask1] - period_n_minus_1
            t2 = t[mask2] - period_n
            biomass1 = total_biomass[mask1]
            biomass2 = total_biomass[mask2]
            
            axes[1, 1].plot(t1, biomass1, '-', linewidth=2, alpha=0.7, 
                          label=f'Period {num_periods-1}', color='blue')
            axes[1, 1].plot(t2, biomass2, '--', linewidth=2, alpha=0.7,
                          label=f'Period {num_periods}', color='orange')
            axes[1, 1].set_xlabel('Time within Period', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Total Biomass', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('Consecutive Periods Comparison', fontsize=12, fontweight='bold')
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == '__main__':
    results, X, t, x_periodic = run_periodic_analysis()
    
    # Plot results
    plot_periodic_convergence(
        X, t, results['period'], results['num_periods_simulated'],
        save_path='PM5/plot9_periodic_convergence.png'
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

