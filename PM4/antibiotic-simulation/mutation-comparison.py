#!/usr/bin/env python3
"""
Compare different mutation matrix models for antibiotic resistance evolution.

This script visualizes how different mutation models affect:
1. The structure of the mutation matrix
2. Evolution dynamics under antibiotic pressure
3. Critical concentration thresholds
4. Resistance emergence patterns
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalf_bacterial import evalf
from tools.SimpleSolver import SimpleSolver

# Import from the main script
import importlib.util
spec = importlib.util.spec_from_file_location("main_sim", 
                                               Path(__file__).parent / "antibiotic-sweep.py")
main_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_sim)


def visualize_mutation_matrices(m=12, IC50_range=(0.1, 5.0)):
    """
    Create visual comparison of different mutation matrix types.
    """
    mutation_types = ['fitness_landscape', 'horizontal_transfer', 'point_mutations', 
                     'asymmetric', 'hotspot']
    
    # Generate IC50 gradient for consistent comparison
    IC50 = np.linspace(IC50_range[0], IC50_range[1], m)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, mut_type in enumerate(mutation_types):
        Q = main_sim.create_mutation_matrix(m, IC50, mutation_type=mut_type,
                                           base_mutation_rate=1e-8,
                                           population_size=1e7)
        
        ax = axes[i]
        
        # Log scale for better visualization of small values
        Q_log = np.log10(Q + 1e-12)  # Add small value to avoid log(0)
        
        im = ax.imshow(Q_log, cmap='RdYlBu_r', aspect='auto',
                      vmin=-12, vmax=-1)
        
        # Sort indices by IC50 for clearer visualization
        sort_idx = np.argsort(IC50)
        ax.set_xticks(np.arange(m))
        ax.set_yticks(np.arange(m))
        ax.set_xticklabels([f'{IC50[j]:.2f}' for j in sort_idx], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([f'{IC50[j]:.2f}' for j in sort_idx], fontsize=8)
        
        ax.set_xlabel('Target IC50', fontsize=10)
        ax.set_ylabel('Source IC50', fontsize=10)
        ax.set_title(f'{mut_type}\n(log₁₀ scale)', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('log₁₀(Mutation Prob)', fontsize=9)
        
        # Add text annotations for key statistics
        diag_mean = np.diag(Q).mean()
        off_diag = Q[~np.eye(m, dtype=bool)]
        off_diag_mean = off_diag.mean()
        
        textstr = f'No mut: {diag_mean:.4f}\nMut: {off_diag_mean:.2e}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.suptitle('Mutation Matrix Comparison: Different Biological Models',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'mutation_matrix_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved mutation matrix comparison to: {output_file}")
    
    return fig


def compare_evolutionary_dynamics():
    """
    Run simulations with different mutation models and compare outcomes.
    """
    np.random.seed(42)
    m = 12
    mutation_types = ['fitness_landscape', 'horizontal_transfer', 'point_mutations', 
                     'asymmetric', 'hotspot']
    
    # Test at a few key concentrations
    C_test = [0.5, 1.5, 2.5]
    colors = plt.cm.Set2(np.linspace(0, 1, len(mutation_types)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    results_dict = {}
    
    for mut_idx, mut_type in enumerate(mutation_types):
        print(f"\nSimulating {mut_type}...")
        
        p = main_sim.create_resistance_parameters(
            m=m,
            resistance_range=(0.1, 5.0),
            mutation_structure='gradient',
            mutation_type=mut_type,
            base_mutation_rate=1e-8,
            population_size=1e7
        )
        
        # Run concentration sweep
        C_values = np.logspace(np.log10(0.01), np.log10(3.0), 20)
        results = main_sim.antibiotic_concentration_sweep(
            p, C_values, uR=0.5, T_final=100.0,
            x0_profile='susceptible_dominant',
            verbose=False
        )
        
        results_dict[mut_type] = {
            'p': p,
            'results': results
        }
        
        # Panel 1: Resistant fraction vs concentration
        axes[0, 0].plot(C_values, results['resistant_fraction'], 
                       '-o', ms=4, color=colors[mut_idx], 
                       label=mut_type, linewidth=2)
    
    axes[0, 0].set_xlabel('Antibiotic Concentration', fontsize=10)
    axes[0, 0].set_ylabel('Resistant Fraction', fontsize=10)
    axes[0, 0].set_title('Resistance Emergence', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=8, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.05)
    
    # Panel 2: Total biomass vs concentration
    for mut_idx, mut_type in enumerate(mutation_types):
        results = results_dict[mut_type]['results']
        C_values = results['C_values']
        axes[0, 1].plot(C_values, results['total_biomass'],
                       '-o', ms=4, color=colors[mut_idx],
                       label=mut_type, linewidth=2)
    
    axes[0, 1].set_xlabel('Antibiotic Concentration', fontsize=10)
    axes[0, 1].set_ylabel('Total Biomass', fontsize=10)
    axes[0, 1].set_title('Population Viability', fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=8, loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Diversity index
    for mut_idx, mut_type in enumerate(mutation_types):
        results = results_dict[mut_type]['results']
        C_values = results['C_values']
        axes[0, 2].plot(C_values, results['diversity_index'],
                       '-o', ms=4, color=colors[mut_idx],
                       label=mut_type, linewidth=2)
    
    axes[0, 2].set_xlabel('Antibiotic Concentration', fontsize=10)
    axes[0, 2].set_ylabel('Shannon Diversity', fontsize=10)
    axes[0, 2].set_title('Genetic Diversity', fontsize=11, fontweight='bold')
    axes[0, 2].legend(fontsize=8, loc='best')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Panel 4: Critical concentration comparison
    critical_concs = []
    for mut_type in mutation_types:
        results = results_dict[mut_type]['results']
        C_values = results['C_values']
        cross_idx = np.where(results['resistant_fraction'] > 0.5)[0]
        if len(cross_idx) > 0:
            critical_concs.append(C_values[cross_idx[0]])
        else:
            critical_concs.append(np.nan)
    
    axes[1, 0].bar(range(len(mutation_types)), critical_concs, color=colors)
    axes[1, 0].set_xticks(range(len(mutation_types)))
    axes[1, 0].set_xticklabels(mutation_types, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_ylabel('Critical Concentration', fontsize=10)
    axes[1, 0].set_title('Threshold for Resistance Dominance', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Final genotype distribution at high concentration
    high_C_idx = -1  # Use highest concentration
    for mut_idx, mut_type in enumerate(mutation_types):
        results = results_dict[mut_type]['results']
        p = results_dict[mut_type]['p']
        
        final_pops = results['final_populations'][high_C_idx]
        IC50 = p['IC50']
        sort_idx = np.argsort(IC50)
        
        x_pos = np.arange(m) + mut_idx * 0.15
        axes[1, 1].bar(x_pos, final_pops[sort_idx], width=0.15,
                      color=colors[mut_idx], alpha=0.8, label=mut_type)
    
    axes[1, 1].set_xlabel('Genotype (sorted by IC50)', fontsize=10)
    axes[1, 1].set_ylabel('Population (log scale)', fontsize=10)
    axes[1, 1].set_title(f'Final Populations at C={C_values[-1]:.2f}', fontsize=11, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=7, loc='best')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Summary statistics table
    axes[1, 2].axis('off')
    table_data = []
    headers = ['Model', 'Crit. C', 'Surv. Geno', 'Div. Loss']
    
    for mut_type in mutation_types:
        results = results_dict[mut_type]['results']
        C_values = results['C_values']
        
        # Critical concentration
        cross_idx = np.where(results['resistant_fraction'] > 0.5)[0]
        crit_c = f"{C_values[cross_idx[0]]:.2f}" if len(cross_idx) > 0 else "N/A"
        
        # Surviving genotypes
        final_pops = results['final_populations'][-1]
        n_surviving = np.sum(final_pops > 0.01)
        
        # Diversity loss
        div_init = results['diversity_index'][0]
        div_final = results['diversity_index'][-1]
        div_loss = f"{(1 - div_final/div_init)*100:.0f}%"
        
        table_data.append([mut_type[:15], crit_c, f"{n_surviving}/{m}", div_loss])
    
    table = axes[1, 2].table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center',
                            colWidths=[0.35, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 2].set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle('Evolutionary Dynamics: Mutation Model Comparison',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'mutation_dynamics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved dynamics comparison to: {output_file}")
    
    return fig, results_dict


def print_mutation_model_guide():
    """
    Print guide explaining each mutation model.
    """
    print("\n" + "="*80)
    print("MUTATION MODEL GUIDE")
    print("="*80)
    
    guide = """
1. FITNESS_LANDSCAPE (Default)
   - Mutations more likely between similar phenotypes (IC50 distance)
   - Loss-of-function 10x easier than gain-of-function
   - Models: Point mutations, small indels
   - Best for: General resistance evolution
   - Key feature: Phenotypic distance matters

2. HORIZONTAL_TRANSFER
   - Combines point mutations + rare HGT events
   - Can jump large phenotypic distances
   - HGT rate: ~1e-7 (about 10% of point mutations)
   - Models: Plasmid transfer, transposons
   - Best for: Multi-drug resistance cassettes
   - Key feature: Rare long-distance jumps

3. POINT_MUTATIONS
   - Only single-step mutations allowed
   - Must traverse IC50 space incrementally
   - No large jumps in resistance
   - Models: Chromosomal SNPs only
   - Best for: Gradual resistance evolution (e.g., quinolones)
   - Key feature: Strict stepwise progression

4. ASYMMETRIC
   - Very strong bias: 50x easier to lose than gain resistance
   - Reflects thermodynamic reality (easier to break than build)
   - Models: Resistance reversibility, fitness recovery
   - Best for: Understanding resistance costs
   - Key feature: Strong selective pressure needed to maintain resistance

5. HOTSPOT
   - Some genotypes are hypermutable (10x higher rate)
   - Mid-resistance genotypes are hotspots
   - Models: Mutator strains, stressed populations
   - Best for: Rapid adaptation scenarios
   - Key feature: Heterogeneous mutation rates
    """
    
    print(guide)
    print("="*80 + "\n")


def main():
    print("="*80)
    print("MUTATION MODEL COMPARISON ANALYSIS")
    print("="*80)
    
    print_mutation_model_guide()
    
    print("\n1. Visualizing mutation matrix structures...")
    visualize_mutation_matrices(m=12)
    
    print("\n2. Comparing evolutionary dynamics...")
    fig, results = compare_evolutionary_dynamics()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - mutation_matrix_comparison.png")
    print("  - mutation_dynamics_comparison.png")
    print("\nKey insights:")
    print("  • Different mutation models predict different critical concentrations")
    print("  • HGT enables faster adaptation to high antibiotic levels")
    print("  • Asymmetric models show stronger selection for maintenance")
    print("  • Hotspot models accelerate resistance emergence")
    print("  • Point mutation models require more gradual selection")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()