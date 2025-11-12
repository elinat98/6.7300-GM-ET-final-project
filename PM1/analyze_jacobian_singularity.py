"""
Analyze where the Jacobian becomes singular and condition number magnitudes.

This script systematically explores parameter space to identify:
1. Regions where Jacobian is singular or near-singular
2. Condition number magnitudes and their biological/physical meaning
3. Intuitive explanations for why singularity occurs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jacobian_tools import evaljacobianf
from evalf_bacterial import evalf


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


def analyze_singularity_regions(p, m=3, n_fixed=None, R_range=(0.001, 5.0), C_range=(0.0, 5.0), 
                                n_points=100, save_path=None):
    """
    Scan (R, C) space to find where Jacobian becomes singular.
    
    Parameters
    ----------
    p : dict
        Model parameters
    m : int
        Number of subpopulations
    n_fixed : array, optional
        Fixed population sizes (if None, uses equal populations)
    R_range : tuple
        (R_min, R_max) for scanning
    C_range : tuple
        (C_min, C_max) for scanning
    n_points : int
        Number of points in each dimension
    save_path : str, optional
        Path to save figure
    """
    if n_fixed is None:
        n_fixed = np.ones(m) * 5.0
    
    R_vals = np.logspace(np.log10(R_range[0]), np.log10(R_range[1]), n_points)
    C_vals = np.linspace(C_range[0], C_range[1], n_points)
    
    cond_grid = np.zeros((len(R_vals), len(C_vals)))
    minsv_grid = np.zeros_like(cond_grid)
    maxsv_grid = np.zeros_like(cond_grid)
    rank_grid = np.zeros_like(cond_grid, dtype=int)
    
    print(f"Scanning {len(R_vals)}×{len(C_vals)} = {len(R_vals)*len(C_vals)} points...")
    
    for i, Rv in enumerate(R_vals):
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(R_vals)} rows")
        for j, Cv in enumerate(C_vals):
            x_try = np.concatenate([n_fixed, np.array([Rv, Cv])])
            try:
                J = evaljacobianf(x_try, p, np.array([0.5, 0.1]))
                s = np.linalg.svd(J, compute_uv=False)
                
                maxsv_grid[i, j] = s[0]
                minsv_grid[i, j] = s[-1]
                rank_grid[i, j] = np.sum(s > 1e-10)  # Effective rank
                
                # Condition number (handle near-zero singular values)
                if s[-1] > 1e-12:
                    cond_grid[i, j] = s[0] / s[-1]
                else:
                    cond_grid[i, j] = np.inf
            except:
                cond_grid[i, j] = np.inf
                minsv_grid[i, j] = 0.0
    
    # Find worst-conditioned regions
    finite_conds = cond_grid[np.isfinite(cond_grid)]
    if len(finite_conds) > 0:
        worst_i, worst_j = np.unravel_index(np.nanargmax(cond_grid), cond_grid.shape)
        worst_R = R_vals[worst_i]
        worst_C = C_vals[worst_j]
        worst_cond = cond_grid[worst_i, worst_j]
        worst_minsv = minsv_grid[worst_i, worst_j]
        
        print(f"\n{'='*70}")
        print("SINGULARITY ANALYSIS RESULTS")
        print(f"{'='*70}")
        print(f"\nWorst-conditioned point:")
        print(f"  R = {worst_R:.4e}, C = {worst_C:.4e}")
        print(f"  Condition number: {worst_cond:.4e}")
        print(f"  Min singular value: {worst_minsv:.4e}")
        print(f"  Max singular value: {maxsv_grid[worst_i, worst_j]:.4e}")
        
        # Find near-singular regions (condition number > threshold)
        threshold = 1e10
        near_singular = cond_grid > threshold
        n_near_singular = np.sum(near_singular)
        print(f"\nNear-singular regions (cond > {threshold:.1e}):")
        print(f"  {n_near_singular} points ({100*n_near_singular/cond_grid.size:.2f}% of space)")
        
        # Statistics
        print(f"\nCondition number statistics:")
        print(f"  Min: {np.nanmin(finite_conds):.4e}")
        print(f"  Median: {np.nanmedian(finite_conds):.4e}")
        print(f"  Max: {np.nanmax(finite_conds):.4e}")
        print(f"  Mean: {np.nanmean(finite_conds):.4e}")
        
        print(f"\nMin singular value statistics:")
        print(f"  Min: {np.nanmin(minsv_grid):.4e}")
        print(f"  Median: {np.nanmedian(minsv_grid):.4e}")
        print(f"  Max: {np.nanmax(minsv_grid):.4e}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Condition number (log scale)
    ax = axes[0, 0]
    finite_mask = np.isfinite(cond_grid)
    if np.any(finite_mask):
        im = ax.contourf(C_vals, R_vals, np.log10(cond_grid + 1e-10), 
                         levels=50, cmap='hot', extend='both')
        ax.set_xlabel('Antibiotic concentration C', fontsize=11)
        ax.set_ylabel('Resource concentration R', fontsize=11)
        ax.set_title('Condition Number log₁₀(κ(J))', fontsize=12, fontweight='bold')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        plt.colorbar(im, ax=ax, label='log₁₀(condition number)')
        # Mark worst point
        if np.isfinite(worst_cond):
            ax.plot(worst_C, worst_R, 'r*', markersize=15, label='Worst conditioned')
            ax.legend()
    
    # Plot 2: Minimum singular value
    ax = axes[0, 1]
    im = ax.contourf(C_vals, R_vals, np.log10(minsv_grid + 1e-15), 
                     levels=50, cmap='viridis', extend='both')
    ax.set_xlabel('Antibiotic concentration C', fontsize=11)
    ax.set_ylabel('Resource concentration R', fontsize=11)
    ax.set_title('Minimum Singular Value log₁₀(σₘᵢₙ)', fontsize=12, fontweight='bold')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='log₁₀(min singular value)')
    
    # Plot 3: Effective rank
    ax = axes[1, 0]
    im = ax.contourf(C_vals, R_vals, rank_grid, levels=np.arange(0, m+3), 
                     cmap='RdYlGn', extend='both')
    ax.set_xlabel('Antibiotic concentration C', fontsize=11)
    ax.set_ylabel('Resource concentration R', fontsize=11)
    ax.set_title(f'Effective Rank (should be {m+2})', fontsize=12, fontweight='bold')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Effective rank')
    
    # Plot 4: Condition number histogram
    ax = axes[1, 1]
    if len(finite_conds) > 0:
        ax.hist(np.log10(finite_conds), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.log10(np.nanmedian(finite_conds)), color='red', 
                   linestyle='--', linewidth=2, label=f'Median: {np.nanmedian(finite_conds):.2e}')
        ax.set_xlabel('log₁₀(Condition Number)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Condition Number Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()
    
    return cond_grid, minsv_grid, R_vals, C_vals


def analyze_extreme_cases(p, m=3):
    """
    Analyze condition numbers at extreme parameter values to understand
    when and why the Jacobian becomes singular.
    """
    print(f"\n{'='*70}")
    print("EXTREME CASE ANALYSIS")
    print(f"{'='*70}\n")
    
    n_base = np.ones(m) * 5.0
    u = np.array([0.5, 0.1])
    
    cases = [
        ("Low resource, no antibiotic", 0.001, 0.0),
        ("Low resource, high antibiotic", 0.001, 5.0),
        ("High resource, no antibiotic", 5.0, 0.0),
        ("High resource, high antibiotic", 5.0, 5.0),
        ("Very low resource", 1e-6, 0.1),
        ("Zero populations", np.zeros(m), 1.0, 0.1),
        ("Extreme antibiotic", np.ones(m)*10.0, 1.0, 10.0),
    ]
    
    results = []
    for case_name, *args in cases:
        if len(args) == 2:
            R, C = args
            n = n_base
        else:
            n, R, C = args
        
        x = np.concatenate([np.asarray(n).ravel(), np.array([R, C])])
        try:
            J = evaljacobianf(x, p, u)
            s = np.linalg.svd(J, compute_uv=False)
            cond_num = s[0] / (s[-1] + 1e-15)
            rank = np.sum(s > 1e-10)
            
            results.append({
                'case': case_name,
                'R': R,
                'C': C,
                'n': n,
                'cond': cond_num,
                'min_sv': s[-1],
                'max_sv': s[0],
                'rank': rank
            })
            
            print(f"{case_name}:")
            print(f"  R={R:.4e}, C={C:.4e}, n={np.asarray(n)}")
            print(f"  Condition number: {cond_num:.4e}")
            print(f"  Min singular value: {s[-1]:.4e}")
            print(f"  Effective rank: {rank}/{m+2}")
            print()
        except Exception as e:
            print(f"{case_name}: ERROR - {e}\n")
    
    return results


def explain_singularity_intuition():
    """
    Print intuitive explanations for why the Jacobian becomes singular.
    """
    print(f"\n{'='*70}")
    print("INTUITIVE EXPLANATION: Why Does the Jacobian Become Singular?")
    print(f"{'='*70}\n")
    
    explanations = [
        ("1. Low Resource (R → 0)", 
         "When R is very small, the Monod term R/(K+R) → 0, so birth rates b → 0. "
         "This makes the ∂f/∂n block nearly zero (no growth), reducing sensitivity. "
         "The system becomes 'stiff' - small changes in populations don't affect dynamics much."),
        
        ("2. High Antibiotic (C >> IC50)", 
         "When C >> IC50, the Hill term → 0, so birth rates → 0. Similar to low resource, "
         "this decouples population dynamics. The system becomes insensitive to population changes."),
        
        ("3. Zero Populations (n → 0)", 
         "When populations are near zero, the coupling terms (n * db/dR, n * db/dC) vanish. "
         "The resource and antibiotic equations become decoupled from population dynamics, "
         "reducing the effective rank of the Jacobian."),
        
        ("4. Extreme Parameter Combinations", 
         "Combinations like R→0 AND C>>IC50 create multiple near-zero blocks simultaneously. "
         "This can cause rank deficiency - the Jacobian loses information about certain directions."),
        
        ("5. Condition Number Magnitudes", 
         "• κ(J) ~ 10²-10³: Well-conditioned, typical for healthy dynamics\n"
         "• κ(J) ~ 10⁴-10⁶: Moderately ill-conditioned, may cause numerical issues\n"
         "• κ(J) ~ 10⁸-10¹²: Poorly conditioned, near-singular, Newton may fail\n"
         "• κ(J) > 10¹²: Effectively singular, system is degenerate"),
        
        ("6. Biological Interpretation", 
         "High condition numbers indicate the system is in a 'brittle' state where:\n"
         "  - Small perturbations can cause large changes (unstable)\n"
         "  - Multiple states are nearly equivalent (non-unique solutions)\n"
         "  - The system is transitioning between regimes (e.g., extinction vs. survival)\n"
         "This makes sense biologically: near extinction or extreme stress, the system "
         "becomes less predictable and more sensitive to numerical errors.")
    ]
    
    for title, explanation in explanations:
        print(f"{title}")
        print(f"  {explanation}\n")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze Jacobian singularity and condition numbers'
    )
    parser.add_argument('--m', type=int, default=3,
                       help='Number of subpopulations (default: 3)')
    parser.add_argument('--n-points', type=int, default=100,
                       help='Number of grid points per dimension (default: 100)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save figure')
    parser.add_argument('--no-extreme', action='store_true',
                       help='Skip extreme case analysis')
    parser.add_argument('--no-explanation', action='store_true',
                       help='Skip intuitive explanation')
    
    args = parser.parse_args()
    
    # Create parameters
    p = default_model_params(m=args.m)
    
    # Run main analysis
    print(f"Analyzing Jacobian singularity for m={args.m} subpopulations...")
    cond_grid, minsv_grid, R_vals, C_vals = analyze_singularity_regions(
        p, m=args.m, n_points=args.n_points, 
        save_path=args.save or f'PM1/jacobian_singularity_analysis_m{args.m}.png'
    )
    
    # Extreme case analysis
    if not args.no_extreme:
        analyze_extreme_cases(p, m=args.m)
    
    # Intuitive explanation
    if not args.no_explanation:
        explain_singularity_intuition()


if __name__ == '__main__':
    main()

