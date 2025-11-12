"""
Visualize Jacobian sparsity pattern for the bacterial evolution model.

This script computes the Jacobian at a given state and visualizes:
1. Sparsity pattern (which elements are non-zero)
2. Magnitude heatmap (color-coded by value)
3. Block structure annotations
4. Sparsity statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jacobian_tools import evaljacobianf
from evalf_bacterial import evalf


def default_model_params(m=3):
    """Create default parameters for testing."""
    # Generate parameters that scale with m
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
    # Ensure Q is properly sized
    if p['Q'].shape[0] != m:
        p['Q'] = np.eye(m)
    return p


def visualize_jacobian_sparsity(J, x, p, save_path=None, show_blocks=True):
    """
    Visualize Jacobian sparsity pattern and structure.
    
    Parameters
    ----------
    J : ndarray
        Jacobian matrix, shape (m+2, m+2)
    x : ndarray
        State vector at which Jacobian was computed
    p : dict
        Model parameters
    save_path : str, optional
        Path to save figure (if None, displays interactively)
    show_blocks : bool
        Whether to annotate block structure
    """
    m = x.size - 2
    N = m + 2
    
    # Compute sparsity statistics
    nnz = np.count_nonzero(J)
    total = J.size
    sparsity = 1.0 - (nnz / total)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 6))
    
    # --- Subplot 1: Sparsity pattern (spy plot) ---
    ax1 = plt.subplot(1, 2, 1)
    # Create binary mask for non-zero elements
    J_mask = (J != 0)
    ax1.spy(J_mask, markersize=4, color='black')
    ax1.set_title(f'Jacobian Sparsity Pattern\n(non-zeros: {nnz}/{total}, sparsity: {sparsity:.1%})', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Column index (state variable)', fontsize=10)
    ax1.set_ylabel('Row index (equation)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add block structure annotations
    if show_blocks and m > 0:
        # Vertical lines
        ax1.axvline(x=m-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Block boundaries')
        ax1.axvline(x=m+1-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        # Horizontal lines
        ax1.axhline(y=m-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.axhline(y=m+1-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Add text annotations for blocks
        ax1.text(m/2, m/2, f'∂f/∂n\n({m}×{m})', 
                ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        ax1.text(m+0.5, m/2, f'∂f/∂R\n({m}×1)', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax1.text(m+1.5, m/2, f'∂f/∂C\n({m}×1)', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax1.text(m/2, m+0.5, f'∂R/∂n\n(1×{m})', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        ax1.text(m+0.5, m+0.5, '∂R/∂R', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        ax1.text(m+1.5, m+0.5, '∂R/∂C', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        ax1.text(m+1.5, m+1.5, '∂C/∂C', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        ax1.legend(loc='upper right', fontsize=8)
    
    # Add axis labels for state variables
    state_labels = [f'n{i+1}' for i in range(m)] + ['R', 'C']
    ax1.set_xticks(range(N))
    ax1.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(N))
    ax1.set_yticklabels(state_labels, fontsize=8)
    
    # --- Subplot 2: Magnitude heatmap (log scale) ---
    ax2 = plt.subplot(1, 2, 2)
    # Use log scale for better visualization, handling zeros
    J_log = np.zeros_like(J)
    J_log[J != 0] = np.log10(np.abs(J[J != 0]))
    J_log[J == 0] = np.nan
    
    im = ax2.imshow(J_log, cmap='viridis', aspect='auto', interpolation='nearest')
    ax2.set_title('Jacobian Magnitude (log₁₀ scale)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Column index (state variable)', fontsize=10)
    ax2.set_ylabel('Row index (equation)', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('log₁₀(|J_ij|)', fontsize=9)
    
    # Add block structure annotations
    if show_blocks and m > 0:
        ax2.axvline(x=m-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=m+1-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=m-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=m+1-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add axis labels
    ax2.set_xticks(range(N))
    ax2.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(N))
    ax2.set_yticklabels(state_labels, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def print_sparsity_statistics(J, x, p):
    """Print detailed sparsity statistics."""
    m = x.size - 2
    N = m + 2
    
    print("="*70)
    print("JACOBIAN SPARSITY ANALYSIS")
    print("="*70)
    print(f"\nMatrix dimensions: {N} × {N} (m = {m} subpopulations + 2 state variables)")
    
    # Overall statistics
    nnz = np.count_nonzero(J)
    total = J.size
    sparsity = 1.0 - (nnz / total)
    density = nnz / total
    
    print(f"\nOverall Statistics:")
    print(f"  Total elements: {total}")
    print(f"  Non-zero elements: {nnz}")
    print(f"  Zero elements: {total - nnz}")
    print(f"  Sparsity: {sparsity:.2%}")
    print(f"  Density: {density:.2%}")
    
    # Block-wise statistics
    if m > 0:
        print(f"\nBlock-wise Statistics:")
        
        # Top-left block: ∂f/∂n (m × m)
        J_nn = J[:m, :m]
        nnz_nn = np.count_nonzero(J_nn)
        print(f"  Block ∂f/∂n ({m}×{m}): {nnz_nn}/{J_nn.size} non-zero ({nnz_nn/J_nn.size:.1%})")
        
        # Top-right blocks: ∂f/∂R and ∂f/∂C (m × 1 each)
        J_nR = J[:m, m]
        J_nC = J[:m, m+1]
        nnz_nR = np.count_nonzero(J_nR)
        nnz_nC = np.count_nonzero(J_nC)
        print(f"  Block ∂f/∂R ({m}×1): {nnz_nR}/{m} non-zero ({nnz_nR/m:.1%})")
        print(f"  Block ∂f/∂C ({m}×1): {nnz_nC}/{m} non-zero ({nnz_nC/m:.1%})")
        
        # Bottom-left: ∂R/∂n (1 × m)
        J_Rn = J[m, :m]
        nnz_Rn = np.count_nonzero(J_Rn)
        print(f"  Block ∂R/∂n (1×{m}): {nnz_Rn}/{m} non-zero ({nnz_Rn/m:.1%})")
        
        # Bottom-right: ∂R/∂R, ∂R/∂C, ∂C/∂R, ∂C/∂C (2 × 2)
        J_RC = J[m:, m:]
        nnz_RC = np.count_nonzero(J_RC)
        print(f"  Block ∂(R,C)/∂(R,C) (2×2): {nnz_RC}/4 non-zero ({nnz_RC/4:.1%})")
    
    # Magnitude statistics
    J_abs = np.abs(J)
    J_nonzero = J_abs[J != 0]
    if len(J_nonzero) > 0:
        print(f"\nMagnitude Statistics (non-zero elements only):")
        print(f"  Min: {J_nonzero.min():.2e}")
        print(f"  Max: {J_nonzero.max():.2e}")
        print(f"  Mean: {J_nonzero.mean():.2e}")
        print(f"  Median: {np.median(J_nonzero):.2e}")
        print(f"  Std: {J_nonzero.std():.2e}")
    
    # Condition number
    try:
        cond_num = np.linalg.cond(J)
        print(f"\nCondition Number: {cond_num:.2e}")
        if cond_num > 1e12:
            print("  WARNING: Matrix is ill-conditioned!")
        elif cond_num > 1e8:
            print("  WARNING: Matrix is poorly conditioned")
        else:
            print("  Matrix is well-conditioned")
    except:
        print("\nCondition Number: Could not compute (matrix may be singular)")
    
    print("="*70)


def main():
    """Main function to run Jacobian sparsity visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize Jacobian sparsity pattern for bacterial evolution model'
    )
    parser.add_argument('--m', type=int, default=3,
                       help='Number of bacterial subpopulations (default: 3)')
    parser.add_argument('--x', type=str, default=None,
                       help='State vector as comma-separated values (default: auto-generated)')
    parser.add_argument('--u', type=str, default='0.5,0.1',
                       help='Input vector [uR, uC] as comma-separated values (default: 0.5,0.1)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save figure (default: display interactively)')
    parser.add_argument('--no-blocks', action='store_true',
                       help='Do not show block structure annotations')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only print statistics, do not create plots')
    
    args = parser.parse_args()
    
    # Create parameters
    p = default_model_params(m=args.m)
    
    # Create state vector
    if args.x is None:
        # Default: equal populations, moderate resource and antibiotic
        x = np.array([10.0] * args.m + [1.0, 0.2])
    else:
        x = np.array([float(v) for v in args.x.split(',')])
        if x.size != args.m + 2:
            raise ValueError(f"State vector must have length {args.m + 2}, got {x.size}")
    
    # Parse input vector
    u = np.array([float(v) for v in args.u.split(',')])
    if u.size != 2:
        raise ValueError("Input vector must have length 2 [uR, uC]")
    
    print(f"Computing Jacobian for m={args.m} subpopulations...")
    print(f"State vector: {x}")
    print(f"Input vector: u = {u}")
    
    # Compute Jacobian
    J = evaljacobianf(x, p, u)
    
    # Print statistics
    print_sparsity_statistics(J, x, p)
    
    # Create visualization
    if not args.stats_only:
        visualize_jacobian_sparsity(J, x, p, save_path=args.save, 
                                   show_blocks=not args.no_blocks)


if __name__ == '__main__':
    main()

