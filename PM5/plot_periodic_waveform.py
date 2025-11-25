#!/usr/bin/env python3
"""
Plot one period of the periodic steady-state waveform.

Shows the evolution of state components over one period [0, T].
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalf_bacterial import evalf
from PM5.model_setup import setup_12_genotype_model
from PM5.forward_euler import forward_euler
from PM5.periodic_steady_state import create_periodic_input, periodic_steady_state_solver
from jacobian_tools import evaljacobianf


def plot_periodic_waveform(x0_periodic, p, eval_u, period, dt, save_path=None):
    """
    Plot one period of the periodic steady-state waveform.
    
    Parameters
    ----------
    x0_periodic : array
        Initial condition for periodic steady state
    p : dict
        Model parameters
    eval_u : callable
        Periodic input function
    period : float
        Period T
    dt : float
        Time step for integration
    save_path : str, optional
        Path to save figure
    """
    # Integrate one period from periodic steady state
    X_period, t_period = forward_euler(
        evalf, x0_periodic, p, eval_u,
        0.0, period, dt, verbose=False
    )
    
    m = 12  # Number of genotypes
    N = len(x0_periodic)  # Total state dimension
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Total biomass over time (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    total_biomass = np.sum(X_period[:m, :], axis=0)
    ax1.plot(t_period, total_biomass, '-', linewidth=2.5, color='darkblue')
    ax1.set_xlabel('Time $t$ (within period)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Biomass $\\sum_i n_i$', fontsize=12, fontweight='bold')
    ax1.set_title('Total Population Evolution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, period])
    
    # Plot 2: Resource R and Antibiotic C (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    R = X_period[m, :]
    C = X_period[m+1, :]
    
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(t_period, R, '-', linewidth=2.5, color='green', label='Resource $R$')
    line2 = ax2_twin.plot(t_period, C, '-', linewidth=2.5, color='red', label='Antibiotic $C$')
    
    ax2.set_xlabel('Time $t$ (within period)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Resource $R$', fontsize=12, fontweight='bold', color='green')
    ax2_twin.set_ylabel('Antibiotic $C$', fontsize=12, fontweight='bold', color='red')
    ax2.set_title('Resource and Antibiotic Concentrations', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, period])
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Plot 3: Selected genotype populations over time (bottom-left)
    # Choose a subset to avoid overcrowding
    ax3 = fig.add_subplot(gs[1, :])
    
    # Select representative genotypes: lowest, middle, highest resistance
    selected_indices = [0, 5, 11]  # First, middle, last genotype
    colors = ['blue', 'orange', 'purple']
    labels = [f'Genotype {i+1}' for i in selected_indices]
    
    for idx, color, label in zip(selected_indices, colors, labels):
        ax3.plot(t_period, X_period[idx, :], '-', linewidth=2, 
                color=color, label=label, alpha=0.8)
    
    ax3.set_xlabel('Time $t$ (within period)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Population $n_i$', fontsize=12, fontweight='bold')
    ax3.set_title('Selected Genotype Populations Over One Period', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, period])
    
    # Plot 4: State components at selected times (bottom-right, top)
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Select representative times
    num_time_samples = 5
    time_indices = np.linspace(0, len(t_period)-1, num_time_samples, dtype=int)
    time_labels = [f'$t = {t_period[i]:.2f}$' for i in time_indices]
    colors_time = plt.cm.viridis(np.linspace(0, 1, num_time_samples))
    
    # Plot all genotype populations at these times
    genotype_indices = np.arange(m)
    
    for i, (t_idx, color, label) in enumerate(zip(time_indices, colors_time, time_labels)):
        ax4.plot(genotype_indices, X_period[:m, t_idx], 'o-', 
                linewidth=1.5, markersize=4, color=color, label=label, alpha=0.7)
    
    ax4.set_xlabel('Genotype Index $i$', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Population $n_i(t)$', fontsize=12, fontweight='bold')
    ax4.set_title('Population Distribution at Selected Times', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(genotype_indices[::2])  # Show every other genotype index
    
    # Plot 5: State vector snapshot at different times (bottom-right, bottom)
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Show full state vector (all components) at a few key times
    # Time 0, pulse start, pulse end, period end
    pulse_duration = 0.1  # From create_periodic_input
    key_times = [0.0, pulse_duration * 0.5, pulse_duration, period * 0.5, period]
    key_indices = [np.argmin(np.abs(t_period - t)) for t in key_times]
    
    # Plot as bar chart or line plot
    state_labels = [f'$n_{i+1}$' for i in range(m)] + ['$R$', '$C$']
    x_positions = np.arange(N)
    
    for t_idx, time_val in zip(key_indices, key_times):
        if time_val <= period:
            ax5.plot(x_positions, X_period[:, t_idx], 'o-', 
                    linewidth=1.5, markersize=3, label=f'$t = {time_val:.2f}$', alpha=0.7)
    
    ax5.set_xlabel('State Component Index', fontsize=12, fontweight='bold')
    ax5.set_ylabel('State Value $x_i$', fontsize=12, fontweight='bold')
    ax5.set_title('Full State Vector at Key Times', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xticks(x_positions[::2])  # Show subset of indices
    ax5.set_xticklabels([state_labels[i] for i in x_positions[::2]], rotation=45, ha='right')
    
    plt.suptitle('Periodic Steady-State Waveform (One Period)', 
                fontsize=15, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def main():
    """Main function to find and plot periodic steady state."""
    print("="*70)
    print("PERIODIC STEADY-STATE WAVEFORM ANALYSIS")
    print("="*70)
    
    # Setup
    p, x0, _ = setup_12_genotype_model()
    
    # Create periodic input
    period = 1.0
    pulse_duration = 0.1
    uC_pulse = 2.0
    eval_u_periodic = create_periodic_input(
        uR_base=0.5,
        uC_pulse=uC_pulse,
        pulse_duration=pulse_duration,
        period=period
    )
    
    # Forward Euler parameters
    dt = 0.01
    
    # Load existing results or compute
    try:
        with open('PM5/periodic_steady_state_results.json', 'r') as f:
            results = json.load(f)
        
        if results.get('x_periodic') is not None:
            x0_periodic = np.array(results['x_periodic'])
            print("Loaded periodic steady state from previous results")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, KeyError):
        print("\n[1/2] Finding periodic steady state using Shooting-Newton...")
        
        # Run transient for initial guess
        from PM5.forward_euler import forward_euler
        X_transient, t_transient = forward_euler(
            evalf, x0, p, eval_u_periodic,
            0.0, 20.0 * period, dt, verbose=False
        )
        x0_guess = X_transient[:, -1]
        
        # Solve for periodic steady state
        x0_periodic, info = periodic_steady_state_solver(
            evalf, evaljacobianf, x0_guess, p, eval_u_periodic,
            period, dt, tol=1e-6, max_iter=30, verbose=True
        )
        
        if not info['converged']:
            print("Warning: Shooting-Newton did not converge. Using best guess.")
    
    # Verify periodic condition
    print("\n[2/2] Verifying periodic condition and plotting...")
    X_verify, t_verify = forward_euler(
        evalf, x0_periodic, p, eval_u_periodic,
        0.0, period, dt, verbose=False
    )
    error_verification = np.linalg.norm(X_verify[:, -1] - x0_periodic, ord=np.inf)
    print(f"  Verification: ||x({period}) - x(0)||_∞ = {error_verification:.6e}")
    
    # Plot waveform
    plot_periodic_waveform(
        x0_periodic, p, eval_u_periodic, period, dt,
        save_path='PM5/plot10_periodic_waveform.png'
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

