"""
Plot evolution of R (resource), C (antibiotic), and all genotypes over time.

This script runs a simulation and creates comprehensive visualizations showing:
1. All genotype populations over time
2. Resource (R) and antibiotic (C) concentrations over time
3. Total biomass and diversity metrics
4. Genotype fractions over time
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalf_bacterial import evalf
from tools.SimpleSolver import SimpleSolver


def default_model_params(m=3):
    """Create default parameters."""
    rmax_base = np.linspace(1.0, 0.6, max(3, m))
    K_base = np.linspace(0.5, 0.3, max(3, m))
    d0_base = np.linspace(0.2, 0.1, max(3, m))
    IC50_base = np.linspace(1.0, 1.2, max(3, m))
    
    p = {
        'Q': np.eye(m),
        'rmax': rmax_base[:m] if m <= len(rmax_base) else np.linspace(1.0, 0.6, m),
        'K': K_base[:m] if m <= len(K_base) else np.linspace(0.5, 0.3, m),
        'alpha': np.full(m, 0.1),
        'd0': d0_base[:m] if m <= len(d0_base) else np.linspace(0.2, 0.1, m),
        'IC50': IC50_base[:m] if m <= len(IC50_base) else np.linspace(1.0, 1.2, m),
        'h': np.full(m, 1.0),
        'kC': 0.05
    }
    if p['Q'].shape[0] != m:
        p['Q'] = np.eye(m)
    return p


def compute_diversity_metrics(n_trajectory):
    """
    Compute diversity metrics over time.
    
    Parameters
    ----------
    n_trajectory : array, shape (m, n_time)
        Population trajectories for all genotypes
    
    Returns
    -------
    shannon_entropy : array
        Shannon entropy (in nats) over time
    simpson_diversity : array
        Simpson diversity index over time
    total_biomass : array
        Total population over time
    """
    n_time = n_trajectory.shape[1]
    shannon_entropy = np.zeros(n_time)
    simpson_diversity = np.zeros(n_time)
    total_biomass = np.sum(n_trajectory, axis=0)
    
    for t in range(n_time):
        n_t = n_trajectory[:, t]
        total = np.sum(n_t)
        if total > 0:
            fractions = n_t / total
            # Shannon entropy (nats)
            shannon_entropy[t] = -np.sum([p * np.log(p) for p in fractions if p > 0.0])
            # Simpson diversity (1 - sum(p_i^2))
            simpson_diversity[t] = 1.0 - np.sum(fractions**2)
        else:
            shannon_entropy[t] = 0.0
            simpson_diversity[t] = 0.0
    
    return shannon_entropy, simpson_diversity, total_biomass


def plot_evolution_over_time(X, t, p, save_path=None, show_fractions=True):
    """
    Create comprehensive plots of evolution over time.
    
    Parameters
    ----------
    X : array, shape (m+2, n_time)
        State trajectory: [n1, n2, ..., nm, R, C] as columns
    t : array
        Time vector
    p : dict
        Model parameters
    save_path : str, optional
        Path to save figure
    show_fractions : bool
        Whether to show genotype fractions plot
    """
    m = X.shape[0] - 2
    n_genotypes = X[:m, :]
    R = X[m, :]
    C = X[m+1, :]
    
    # Compute diversity metrics
    shannon_entropy, simpson_diversity, total_biomass = compute_diversity_metrics(n_genotypes)
    
    # Determine number of subplots
    n_plots = 4 if show_fractions else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
    
    # --- Plot 1: Genotype populations ---
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 0.9, m))
    for i in range(m):
        ax.plot(t, n_genotypes[i, :], label=f'Genotype {i+1} (IC50={p["IC50"][i]:.2f})', 
                color=colors[i], linewidth=2, alpha=0.8)
    ax.set_ylabel('Population Density', fontsize=11, fontweight='bold')
    ax.set_title('Genotype Populations Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=min(3, m))
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # --- Plot 2: Resource and Antibiotic ---
    ax = axes[1]
    ax.plot(t, R, label='R (Resource)', color='green', linewidth=2.5, linestyle='-')
    ax.plot(t, C, label='C (Antibiotic)', color='red', linewidth=2.5, linestyle='--')
    ax.set_ylabel('Concentration', fontsize=11, fontweight='bold')
    ax.set_title('Resource and Antibiotic Concentrations Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 3: Total Biomass and Diversity ---
    ax = axes[2]
    ax2_twin = ax.twinx()
    
    # Total biomass on left axis
    line1 = ax.plot(t, total_biomass, label='Total Biomass', color='blue', 
                    linewidth=2.5, linestyle='-')
    ax.set_ylabel('Total Biomass', fontsize=11, fontweight='bold', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # Diversity metrics on right axis
    line2 = ax2_twin.plot(t, shannon_entropy, label='Shannon Entropy', 
                          color='orange', linewidth=2, linestyle='--', alpha=0.8)
    line3 = ax2_twin.plot(t, simpson_diversity, label='Simpson Diversity', 
                          color='purple', linewidth=2, linestyle=':', alpha=0.8)
    ax2_twin.set_ylabel('Diversity Index', fontsize=11, fontweight='bold', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    ax.set_title('Total Biomass and Diversity Metrics Over Time', fontsize=12, fontweight='bold')
    
    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 4: Genotype fractions (optional) ---
    if show_fractions:
        ax = axes[3]
        for i in range(m):
            fractions = n_genotypes[i, :] / (total_biomass + 1e-10)
            ax.plot(t, fractions, label=f'Genotype {i+1}', 
                    color=colors[i], linewidth=2, alpha=0.8)
        ax.set_ylabel('Fraction', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_title('Genotype Fractions Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=min(3, m))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        axes[2].set_xlabel('Time', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function to run simulation and create plots."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot evolution of R, C, and genotypes over time'
    )
    parser.add_argument('--m', type=int, default=3,
                       help='Number of subpopulations (default: 3)')
    parser.add_argument('--NumIter', type=int, default=400,
                       help='Number of time steps (default: 400)')
    parser.add_argument('--w', type=float, default=0.01,
                       help='Time step size (default: 0.01)')
    parser.add_argument('--x0', type=str, default=None,
                       help='Initial state as comma-separated values (default: auto-generated)')
    parser.add_argument('--u', type=str, default='0.5,0.1',
                       help='Input vector [uR, uC] as comma-separated values (default: 0.5,0.1)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save figure (default: display interactively)')
    parser.add_argument('--no-fractions', action='store_true',
                       help='Do not show genotype fractions plot')
    
    args = parser.parse_args()
    
    # Create parameters
    p = default_model_params(m=args.m)
    
    # Create initial state
    if args.x0 is None:
        x0 = np.array([10.0] * args.m + [1.0, 0.2]).reshape((-1, 1))
    else:
        x0_vals = np.array([float(v) for v in args.x0.split(',')])
        if x0_vals.size != args.m + 2:
            raise ValueError(f"Initial state must have length {args.m + 2}, got {x0_vals.size}")
        x0 = x0_vals.reshape((-1, 1))
    
    # Parse input vector
    u_vals = np.array([float(v) for v in args.u.split(',')])
    if u_vals.size != 2:
        raise ValueError("Input vector must have length 2 [uR, uC]")
    
    def eval_u(t):
        return u_vals
    
    print(f"Running simulation with m={args.m} subpopulations...")
    print(f"  Initial state: {x0.ravel()}")
    print(f"  Input vector: u = {u_vals}")
    print(f"  Time steps: {args.NumIter}, step size: {args.w}")
    print(f"  Total simulation time: {args.NumIter * args.w:.2f}")
    
    # Run simulation
    X, t = SimpleSolver(evalf, x0, p, eval_u, NumIter=args.NumIter, w=args.w, visualize=False)
    
    print(f"\nSimulation complete!")
    print(f"  Final state: {X[:, -1].ravel()}")
    print(f"  Final total biomass: {np.sum(X[:args.m, -1]):.4f}")
    print(f"  Final R: {X[args.m, -1]:.4f}, C: {X[args.m+1, -1]:.4f}")
    
    # Create plots
    save_path = args.save or f'PM1/evolution_over_time_m{args.m}.png'
    plot_evolution_over_time(X, t, p, save_path=save_path, 
                            show_fractions=not args.no_fractions)
    
    print(f"\nVisualization complete!")


if __name__ == '__main__':
    main()

