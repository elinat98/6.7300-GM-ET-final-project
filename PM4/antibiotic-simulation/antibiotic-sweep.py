#!/usr/bin/env python3
"""
Antibiotic Resistance Evolution Simulation with Concentration Sweep

This script simulates bacterial population dynamics across multiple genotypes
with varying antibiotic resistance levels. It performs a parameter sweep over
antibiotic concentration to analyze:
- Population dynamics of resistant vs susceptible genotypes
- Emergence and dominance of resistant strains
- Critical antibiotic concentrations for selection pressure

Based on clinical studies showing 5-20 genotypes typically present in
resistant bacterial infections (see Pseudomonas aeruginosa, E. coli studies).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ensure repo root on sys.path (go up one level from PM4 to access evalf_bacterial)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalf_bacterial import evalf
from tools.SimpleSolver import SimpleSolver


def create_mutation_matrix(m, IC50, mutation_type='fitness_landscape', 
                          base_mutation_rate=1e-8, population_size=1e7):
    """
    Create biologically realistic mutation matrix Q.
    
    Parameters
    ----------
    m : int
        Number of genotypes
    IC50 : array
        IC50 values for each genotype (resistance levels)
    mutation_type : str
        Type of mutation structure:
        - 'fitness_landscape': Based on phenotypic distance in IC50 space
        - 'horizontal_transfer': Includes rare HGT events to distant genotypes
        - 'point_mutations': Only small IC50 changes (single mutations)
        - 'asymmetric': Easier to lose resistance than gain it
        - 'hotspot': Some genotypes are mutation hotspots
    base_mutation_rate : float
        Per-cell per-division mutation rate (typical: 1e-8 to 1e-10)
    population_size : float
        Effective population size (affects mutation supply)
    
    Returns
    -------
    Q : array (m, m)
        Mutation probability matrix (rows sum to 1)
    
    Notes
    -----
    Biologically realistic features:
    - Mutation rates scale with genetic/phenotypic distance
    - Asymmetric: loss-of-function easier than gain-of-function
    - Rare long-distance jumps (HGT, large deletions)
    - Some genotypes are mutational hotspots
    """
    
    Q = np.zeros((m, m))
    
    if mutation_type == 'fitness_landscape':
        # Mutations are more likely between genetically similar strains
        # Use IC50 distance as proxy for genetic distance
        IC50_normalized = (IC50 - IC50.min()) / (IC50.max() - IC50.min() + 1e-10)
        
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                # Phenotypic distance
                delta_IC50 = np.abs(IC50_normalized[i] - IC50_normalized[j])
                
                # Mutation probability decreases exponentially with distance
                # Single point mutation: delta_IC50 ~ 0.1-0.3
                # Multiple mutations: delta_IC50 > 0.5
                mutation_prob = base_mutation_rate * np.exp(-5.0 * delta_IC50)
                
                # Asymmetry: easier to lose resistance (break function) than gain it
                if IC50[j] < IC50[i]:  # losing resistance
                    mutation_prob *= 10.0  # 10x easier to break than to build
                
                Q[i, j] = mutation_prob
        
        # Scale by population size (more cells = more mutations per generation)
        Q *= population_size
        
    elif mutation_type == 'horizontal_transfer':
        # Combines point mutations + rare HGT events
        IC50_normalized = (IC50 - IC50.min()) / (IC50.max() - IC50.min() + 1e-10)
        
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                delta_IC50 = np.abs(IC50_normalized[i] - IC50_normalized[j])
                
                # Point mutations (short distance)
                point_mut_prob = base_mutation_rate * np.exp(-5.0 * delta_IC50)
                
                # HGT (long distance, rare, but uniform in distance)
                # Typical HGT rate: ~1e-7 per cell per generation
                hgt_prob = 1e-7 * (1.0 if IC50[j] > IC50[i] else 0.1)
                
                mutation_prob = point_mut_prob + hgt_prob
                
                # Asymmetry
                if IC50[j] < IC50[i]:
                    mutation_prob *= 5.0
                
                Q[i, j] = mutation_prob * population_size
        
    elif mutation_type == 'point_mutations':
        # Only single-step mutations (small IC50 changes)
        # Sort by IC50 to define "neighbors"
        sort_idx = np.argsort(IC50)
        IC50_sorted = IC50[sort_idx]
        
        # Build mutation matrix in sorted space
        Q_sorted = np.zeros((m, m))
        single_step_rate = base_mutation_rate * population_size * 10
        
        for i in range(m):
            # Can only mutate to neighbors in IC50 space
            if i > 0:  # to lower resistance
                Q_sorted[i, i-1] = single_step_rate * 5.0  # easier to lose
            if i < m-1:  # to higher resistance
                Q_sorted[i, i+1] = single_step_rate
        
        # Transform back to original ordering
        inv_idx = np.argsort(sort_idx)
        Q = Q_sorted[inv_idx, :][:, inv_idx]
        
    elif mutation_type == 'asymmetric':
        # Strong asymmetry: very easy to lose resistance, hard to gain
        IC50_normalized = (IC50 - IC50.min()) / (IC50.max() - IC50.min() + 1e-10)
        
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                delta_IC50 = np.abs(IC50_normalized[i] - IC50_normalized[j])
                mutation_prob = base_mutation_rate * np.exp(-3.0 * delta_IC50)
                
                # Very strong asymmetry
                if IC50[j] < IC50[i]:  # losing resistance
                    mutation_prob *= 50.0  # 50x easier to lose
                else:  # gaining resistance
                    mutation_prob *= 0.1  # 10x harder to gain
                
                Q[i, j] = mutation_prob * population_size
        
    elif mutation_type == 'hotspot':
        # Some genotypes are mutational hotspots (e.g., due to hypermutable loci)
        IC50_normalized = (IC50 - IC50.min()) / (IC50.max() - IC50.min() + 1e-10)
        
        # Identify hotspot genotypes (middle-resistance: most evolvable)
        hotspot_mask = (IC50_normalized > 0.3) & (IC50_normalized < 0.7)
        
        for i in range(m):
            hotspot_multiplier = 10.0 if hotspot_mask[i] else 1.0
            
            for j in range(m):
                if i == j:
                    continue
                delta_IC50 = np.abs(IC50_normalized[i] - IC50_normalized[j])
                mutation_prob = base_mutation_rate * np.exp(-5.0 * delta_IC50)
                
                if IC50[j] < IC50[i]:
                    mutation_prob *= 10.0
                
                Q[i, j] = mutation_prob * population_size * hotspot_multiplier
    
    else:
        raise ValueError(f"Unknown mutation_type: {mutation_type}")
    
    # Ensure rows sum to <= 1 (can't have more mutations than cells)
    # Clip very high mutation rates
    row_sums = Q.sum(axis=1)
    max_mut_rate = 0.05  # Max 5% of cells mutate per generation
    for i in range(m):
        if row_sums[i] > max_mut_rate:
            Q[i, :] *= max_mut_rate / row_sums[i]
    
    # Set diagonal: probability of no mutation
    for i in range(m):
        Q[i, i] = 1.0 - Q[i, :].sum()
    
    # Final normalization (ensure valid probability distribution)
    Q = Q / Q.sum(axis=1, keepdims=True)
    
    return Q


def create_resistance_parameters(m=12, resistance_range=(0.1, 5.0), 
                                mutation_structure='gradient',
                                mutation_type='fitness_landscape',
                                base_mutation_rate=1e-8,
                                population_size=1e7):
    """
    Create parameters for m genotypes with varying antibiotic resistance.
    
    Parameters
    ----------
    m : int
        Number of genotypes (default 12, representing moderate diversity)
    resistance_range : tuple
        (min_IC50, max_IC50) for resistance spectrum
    mutation_structure : str
        'gradient' - smooth increase in resistance
        'clusters' - distinct resistant and susceptible groups
        'sparse' - mostly susceptible with few highly resistant
    mutation_type : str
        Type of mutation matrix (see create_mutation_matrix for options)
        Default: 'fitness_landscape'
    base_mutation_rate : float
        Per-cell per-generation mutation rate (default 1e-8)
    population_size : float
        Effective population size (default 1e7)
    
    Returns
    -------
    p : dict
        Parameter dictionary for bacterial model
    """
    
    # Base parameters (similar growth rates, slight variations)
    base_rmax = 1.0
    base_K = 0.5
    base_alpha = 0.1
    base_d0 = 0.15
    
    if mutation_structure == 'gradient':
        # Smooth gradient from susceptible to resistant
        IC50 = np.linspace(resistance_range[0], resistance_range[1], m)
        # Fitness cost: more resistant = slightly slower growth
        rmax = base_rmax * (1.0 - 0.15 * np.linspace(0, 1, m))
        d0 = base_d0 * (1.0 + 0.2 * np.linspace(0, 1, m))
        
    elif mutation_structure == 'clusters':
        # Two main clusters: susceptible and resistant
        n_susceptible = m // 3
        n_resistant = m - n_susceptible
        IC50_susceptible = np.linspace(resistance_range[0], 0.5, n_susceptible)
        IC50_resistant = np.linspace(2.0, resistance_range[1], n_resistant)
        IC50 = np.concatenate([IC50_susceptible, IC50_resistant])
        
        rmax = np.concatenate([
            base_rmax * np.ones(n_susceptible),
            base_rmax * 0.85 * np.ones(n_resistant)  # resistance cost
        ])
        d0 = np.concatenate([
            base_d0 * np.ones(n_susceptible),
            base_d0 * 1.2 * np.ones(n_resistant)
        ])
        
    elif mutation_structure == 'sparse':
        # Mostly susceptible with a few highly resistant mutants
        n_resistant = max(2, m // 5)
        n_susceptible = m - n_resistant
        IC50 = np.concatenate([
            np.linspace(resistance_range[0], 0.5, n_susceptible),
            np.linspace(3.0, resistance_range[1], n_resistant)
        ])
        rmax = np.concatenate([
            base_rmax * np.ones(n_susceptible),
            base_rmax * 0.80 * np.ones(n_resistant)
        ])
        d0 = np.concatenate([
            base_d0 * np.ones(n_susceptible),
            base_d0 * 1.25 * np.ones(n_resistant)
        ])
    
    else:
        raise ValueError(f"Unknown mutation_structure: {mutation_structure}")
    
    # Create biologically realistic mutation matrix Q
    Q = create_mutation_matrix(m, IC50, 
                               mutation_type=mutation_type,
                               base_mutation_rate=base_mutation_rate,
                               population_size=population_size)
    
    p = {
        'Q': Q,
        'rmax': rmax,
        'K': base_K * np.ones(m),
        'alpha': base_alpha * np.ones(m),
        'd0': d0,
        'IC50': IC50,
        'h': np.ones(m) * 2.0,  # Hill coefficient (cooperativity)
        'kC': 0.05
    }
    
    return p


def antibiotic_concentration_sweep(p, C_values, uR=0.5, T_final=100.0, 
                                   x0_profile='uniform', verbose=True):
    """
    Sweep over antibiotic concentrations and simulate population dynamics.
    
    Parameters
    ----------
    p : dict
        Parameter dictionary
    C_values : array-like
        Antibiotic concentrations to test
    uR : float
        Resource influx rate
    T_final : float
        Simulation time for each concentration
    x0_profile : str
        Initial population profile ('uniform', 'susceptible_dominant', 'resistant_seed')
    
    Returns
    -------
    results : dict
        Simulation results for each concentration
    """
    m = p['rmax'].size
    results = {
        'C_values': np.array(C_values),
        'final_populations': [],
        'trajectories': [],
        'total_biomass': [],
        'resistant_fraction': [],
        'diversity_index': []
    }
    
    # Initial condition
    if x0_profile == 'uniform':
        n0 = np.ones(m) * 5.0
    elif x0_profile == 'susceptible_dominant':
        n0 = np.zeros(m)
        susceptible_idx = np.argsort(p['IC50'])[:m//2]
        n0[susceptible_idx] = 10.0
        n0[susceptible_idx[0]] = 20.0  # one dominant
    elif x0_profile == 'resistant_seed':
        n0 = np.ones(m) * 8.0
        resistant_idx = np.argsort(p['IC50'])[-2:]
        n0[resistant_idx] = 0.5  # seed small resistant population
    else:
        n0 = np.ones(m) * 5.0
    
    R0 = 1.0
    C0 = 0.0
    x0 = np.concatenate([n0, [R0, C0]]).reshape((-1, 1))
    
    for i, C_level in enumerate(C_values):
        if verbose:
            print(f"Simulating C = {C_level:.4f} ({i+1}/{len(C_values)})")
        
        # Define input function with constant antibiotic
        def eval_u(t):
            return np.array([uR, C_level])
        
        # Simulate
        NumIter = int(T_final / 0.02)
        X, t = SimpleSolver(evalf, x0, p, eval_u, NumIter=NumIter, 
                           w=0.02, visualize=False)
        
        # Extract final state and metrics
        n_final = X[:m, -1]
        total_n = np.sum(n_final)
        
        # Resistant fraction (IC50 > median)
        median_IC50 = np.median(p['IC50'])
        resistant_mask = p['IC50'] > median_IC50
        resistant_frac = np.sum(n_final[resistant_mask]) / max(total_n, 1e-10)
        
        # Shannon diversity index
        freq = n_final / max(total_n, 1e-10)
        freq = freq[freq > 1e-10]
        diversity = -np.sum(freq * np.log(freq)) if len(freq) > 0 else 0.0
        
        results['final_populations'].append(n_final)
        results['trajectories'].append(X)
        results['total_biomass'].append(total_n)
        results['resistant_fraction'].append(resistant_frac)
        results['diversity_index'].append(diversity)
        
        # Update initial condition for next simulation (continuity)
        x0 = X[:, -1].reshape((-1, 1))
    
    # Convert lists to arrays
    results['final_populations'] = np.array(results['final_populations'])
    results['total_biomass'] = np.array(results['total_biomass'])
    results['resistant_fraction'] = np.array(results['resistant_fraction'])
    results['diversity_index'] = np.array(results['diversity_index'])
    
    return results


def plot_sweep_results(results, p, output_dir='.'):
    """
    Create comprehensive visualization of sweep results with improved spacing.
    """
    C_values = results['C_values']
    m = p['rmax'].size
    IC50 = p['IC50']
    
    # Sort genotypes by IC50 for visualization
    sort_idx = np.argsort(IC50)
    IC50_sorted = IC50[sort_idx]
    
    fig = plt.figure(figsize=(20, 14))  # Increased size
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.4,
                  width_ratios=[1.2, 1, 0.8])  # Give more space to heatmap, less to biomass panel
    
    # --- Panel 1: Heatmap of genotype populations vs concentration ---
    ax1 = fig.add_subplot(gs[0, 0])  # Changed from gs[0, :2] to gs[0, 0]
    pop_matrix = results['final_populations'][:, sort_idx].T
    pop_matrix_log = np.log10(pop_matrix + 1e-6)
    
    im1 = ax1.imshow(pop_matrix_log, aspect='auto', cmap='viridis', 
                     extent=[C_values[0], C_values[-1], 0, m],
                     origin='lower', interpolation='bilinear')
    ax1.set_xlabel('Antibiotic Concentration (C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Genotype (sorted by IC50)', fontsize=12, fontweight='bold')
    ax1.set_title('Population Heatmap: log₁₀(Genotype Density)', fontsize=13, fontweight='bold', pad=10)
    
    # Colorbar on the right side
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('log₁₀(cells/mL)', fontsize=10)
    cbar1.ax.tick_params(labelsize=9)
    ax1.tick_params(labelsize=9)
    
    # Add IC50 reference line on the plot itself (not as second y-axis)
    # Overlay IC50 values as a semi-transparent line
    ic50_norm = (IC50_sorted - IC50_sorted.min()) / (IC50_sorted.max() - IC50_sorted.min())
    ic50_scaled = ic50_norm * (C_values[-1] - C_values[0]) * 0.8 + C_values[0]
    ax1.plot(ic50_scaled, np.arange(m), 'r--', alpha=0.7, linewidth=2.5, label='IC50 profile')
    
    # Add text annotation instead of second axis
    ax1.text(0.98, 0.02, 'Red dashed: IC50 profile', 
            transform=ax1.transAxes, ha='right', va='bottom',
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # --- Panel 2: Total biomass and resistant fraction ---
    ax2 = fig.add_subplot(gs[0, 1])  # Changed to column 1 (middle column)
    ax2_twin = ax2.twinx()
    
    ln1 = ax2.plot(C_values, results['total_biomass'], 'b-o', ms=5, 
                   label='Total Biomass', linewidth=2.5, markeredgecolor='darkblue', markeredgewidth=0.5)
    ax2.set_xlabel('Antibiotic Conc. (C)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Biomass', fontsize=11, color='b', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b', labelsize=9)
    ax2.tick_params(axis='x', labelsize=9)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    ln2 = ax2_twin.plot(C_values, results['resistant_fraction'], 'r-s', ms=5,
                        label='Resistant Fraction', linewidth=2.5, markeredgecolor='darkred', markeredgewidth=0.5)
    ax2_twin.set_ylabel('Resistant Fraction', fontsize=11, color='r', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='r', labelsize=9)
    ax2_twin.set_ylim(-0.05, 1.1)
    
    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='center left', fontsize=9, framealpha=0.9)
    ax2.set_title('Biomass & Resistance', fontsize=12, fontweight='bold', pad=8)
    
    # --- Panel 2b: Summary Statistics Box ---
    ax2b = fig.add_subplot(gs[0, 2])
    ax2b.axis('off')
    
    # Calculate key statistics
    total_biomass_max = results['total_biomass'].max()
    total_biomass_min = results['total_biomass'].min()
    biomass_reduction = (1 - total_biomass_min/total_biomass_max) * 100
    
    resistant_initial = results['resistant_fraction'][0]
    resistant_final = results['resistant_fraction'][-1]
    
    diversity_max = results['diversity_index'].max()
    diversity_min = results['diversity_index'].min()
    diversity_loss = (1 - diversity_min/diversity_max) * 100
    
    # Find critical concentration
    cross_idx = np.where(results['resistant_fraction'] > 0.5)[0]
    if len(cross_idx) > 0:
        C_critical = C_values[cross_idx[0]]
        crit_text = f'{C_critical:.2f}'
    else:
        crit_text = '>6.00'
    
    # Surviving genotypes
    final_pops = results['final_populations'][-1]
    n_surviving = np.sum(final_pops > 0.01)
    
    # Create text summary
    stats_text = f"""KEY STATISTICS
    
Population Dynamics:
  • Max biomass: {total_biomass_max:.1f}
  • Min biomass: {total_biomass_min:.1f}
  • Reduction: {biomass_reduction:.1f}%

Resistance Evolution:
  • Initial resistant: {resistant_initial*100:.1f}%
  • Final resistant: {resistant_final*100:.1f}%
  • Critical C: {crit_text}

Genetic Diversity:
  • Max diversity: {diversity_max:.2f}
  • Min diversity: {diversity_min:.2f}
  • Loss: {diversity_loss:.1f}%
  • Surviving: {n_surviving}/{m}
    """
    
    ax2b.text(0.05, 0.95, stats_text, transform=ax2b.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=1))
    ax2b.set_title('Summary', fontsize=12, fontweight='bold', pad=8)
    
    # --- Panel 3: Diversity index ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(C_values, results['diversity_index'], 'g-^', ms=6, linewidth=2.5,
            markeredgecolor='darkgreen', markeredgewidth=0.5)
    ax3.set_xlabel('Antibiotic Conc. (C)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Shannon Diversity Index', fontsize=11, fontweight='bold')
    ax3.set_title('Population Diversity', fontsize=12, fontweight='bold', pad=8)
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    ax3.tick_params(labelsize=9)
    
    # Highlight diversity collapse
    max_div = np.max(results['diversity_index'])
    collapse_threshold = 0.3 * max_div
    collapse_idx = np.where(results['diversity_index'] < collapse_threshold)[0]
    if len(collapse_idx) > 0:
        ax3.axvspan(C_values[collapse_idx[0]], C_values[-1], 
                   alpha=0.15, color='red', label='Diversity collapse')
        ax3.legend(fontsize=9, loc='best', framealpha=0.9)
    
    # --- Panel 4: Genotype distribution at key concentrations ---
    ax4 = fig.add_subplot(gs[1, 1:])
    
    # Select 4 representative concentrations
    n_samples = min(4, len(C_values))
    sample_idx = np.linspace(0, len(C_values)-1, n_samples, dtype=int)
    
    x_pos = np.arange(m)
    width = 0.75 / n_samples
    colors_conc = cm.plasma(np.linspace(0.2, 0.9, n_samples))
    
    for i, idx in enumerate(sample_idx):
        offset = (i - n_samples/2 + 0.5) * width
        pops = results['final_populations'][idx, sort_idx]
        ax4.bar(x_pos + offset, pops, width, label=f'C={C_values[idx]:.2f}',
               color=colors_conc[i], alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax4.set_xlabel('Genotype (sorted by IC50)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Population Density', fontsize=11, fontweight='bold')
    ax4.set_title('Genotype Distribution at Selected Concentrations', fontsize=12, fontweight='bold', pad=8)
    ax4.set_xticks(x_pos[::2])
    ax4.set_xticklabels([f'{i}' for i in x_pos[::2]], fontsize=9)
    ax4.legend(fontsize=9, ncol=2, loc='upper right', framealpha=0.9)
    ax4.set_yscale('log')
    ax4.set_ylim(1e-3, 100)
    ax4.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax4.tick_params(labelsize=9)
    
    # --- Panel 5: IC50 distribution of population ---
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Weighted histogram of IC50 by population at different concentrations
    for i, idx in enumerate(sample_idx):
        pops = results['final_populations'][idx]
        weights = pops / np.sum(pops) if np.sum(pops) > 0 else pops
        ax5.hist(IC50, bins=15, weights=weights, alpha=0.6, 
                label=f'C={C_values[idx]:.2f}', color=colors_conc[i],
                edgecolor='black', linewidth=0.5)
    
    ax5.set_xlabel('IC50 (Resistance Level)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Weighted Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Population-Weighted IC50 Distribution', fontsize=12, fontweight='bold', pad=8)
    ax5.legend(fontsize=9, loc='best', framealpha=0.9)
    ax5.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax5.tick_params(labelsize=9)
    
    # --- Panel 6: Time series at critical concentration ---
    ax6 = fig.add_subplot(gs[2, 1:])
    
    # Find concentration where resistant fraction crosses 0.5
    cross_idx = np.where(results['resistant_fraction'] > 0.5)[0]
    if len(cross_idx) > 0:
        critical_idx = cross_idx[0]
    else:
        critical_idx = len(C_values) // 2
    
    X_crit = results['trajectories'][critical_idx]
    t_crit = np.linspace(0, 100, X_crit.shape[1])
    
    # Plot top 5 genotypes by final population
    top_geno_idx = np.argsort(X_crit[:m, -1])[-5:]
    colors_geno = cm.tab10(np.linspace(0, 0.9, 5))
    
    for i, geno_idx in enumerate(top_geno_idx):
        ax6.plot(t_crit, X_crit[geno_idx, :], linewidth=2.5, 
                color=colors_geno[i], alpha=0.8,
                label=f'Genotype {geno_idx} (IC50={IC50[geno_idx]:.2f})')
    
    ax6.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Population Density', fontsize=11, fontweight='bold')
    ax6.set_title(f'Dynamics at C={C_values[critical_idx]:.2f} (Critical Transition)', 
                 fontsize=12, fontweight='bold', pad=8)
    ax6.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    ax6.set_yscale('log')
    ax6.set_ylim(1e-1, 20)
    ax6.grid(True, alpha=0.3, linewidth=0.5)
    ax6.tick_params(labelsize=9)
    
    plt.suptitle('Antibiotic Resistance Evolution: Concentration Sweep Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = Path(output_dir) / 'antibiotic_resistance_sweep.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    
    return fig


def print_summary_statistics(results, p):
    """
    Print summary statistics from the sweep.
    """
    C_values = results['C_values']
    
    print("\n" + "="*70)
    print("ANTIBIOTIC RESISTANCE SWEEP - SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nModel Configuration:")
    print(f"  Number of genotypes: {p['rmax'].size}")
    print(f"  IC50 range: [{p['IC50'].min():.3f}, {p['IC50'].max():.3f}]")
    print(f"  Concentration range tested: [{C_values.min():.3f}, {C_values.max():.3f}]")
    
    print(f"\nPopulation Dynamics:")
    print(f"  Max total biomass: {results['total_biomass'].max():.2f}")
    print(f"  Min total biomass: {results['total_biomass'].min():.2f}")
    print(f"  Biomass reduction: {(1 - results['total_biomass'].min()/results['total_biomass'].max())*100:.1f}%")
    
    print(f"\nResistance Evolution:")
    print(f"  Initial resistant fraction: {results['resistant_fraction'][0]:.3f}")
    print(f"  Final resistant fraction: {results['resistant_fraction'][-1]:.3f}")
    
    # Find critical concentration (where resistant fraction > 0.5)
    critical_idx = np.where(results['resistant_fraction'] > 0.5)[0]
    if len(critical_idx) > 0:
        C_critical = C_values[critical_idx[0]]
        print(f"  Critical concentration (50% resistant): C = {C_critical:.3f}")
    else:
        print(f"  Critical concentration not reached in tested range")
    
    print(f"\nGenotype Diversity:")
    print(f"  Max diversity index: {results['diversity_index'].max():.3f}")
    print(f"  Min diversity index: {results['diversity_index'].min():.3f}")
    print(f"  Diversity at highest C: {results['diversity_index'][-1]:.3f}")
    
    # Surviving genotypes at highest concentration
    final_pops = results['final_populations'][-1]
    surviving = np.sum(final_pops > 0.01)
    print(f"  Surviving genotypes (>0.01): {surviving}/{p['rmax'].size}")
    
    # Dominant genotype characteristics
    dominant_idx = np.argmax(final_pops)
    print(f"\nDominant Genotype at Highest Concentration:")
    print(f"  Genotype index: {dominant_idx}")
    print(f"  IC50: {p['IC50'][dominant_idx]:.3f}")
    print(f"  Growth rate (rmax): {p['rmax'][dominant_idx]:.3f}")
    print(f"  Death rate (d0): {p['d0'][dominant_idx]:.3f}")
    print(f"  Population: {final_pops[dominant_idx]:.2f}")
    
    print("\n" + "="*70 + "\n")


def main():
    """
    Main simulation driver.
    """
    print("="*70)
    print("ANTIBIOTIC RESISTANCE EVOLUTION SIMULATION")
    print("="*70)
    print("\nBased on clinical studies showing 5-20 genotypes in resistant")
    print("bacterial infections (Pseudomonas aeruginosa, E. coli, etc.)\n")
    
    # Configuration
    np.random.seed(42)
    m = 12  # Number of genotypes (realistic for clinical infections)
    
    # Create parameters with gradient resistance structure
    print(f"Creating model with {m} genotypes...")
    
    # Choose mutation type - try different ones!
    # Options: 'fitness_landscape', 'horizontal_transfer', 'point_mutations', 
    #          'asymmetric', 'hotspot'
    mutation_type = 'fitness_landscape'  # CHANGE THIS TO TEST DIFFERENT MODELS
    
    p = create_resistance_parameters(
        m=m, 
        resistance_range=(0.1, 5.0),
        mutation_structure='gradient',  # Try 'clusters' or 'sparse' too
        mutation_type=mutation_type,
        base_mutation_rate=1e-8,  # Typical bacterial mutation rate
        population_size=1e7  # 10 million cells
    )
    
    print(f"\nMutation model: {mutation_type}")
    print(f"IC50 values: {p['IC50']}")
    print(f"Growth rates (rmax): {p['rmax']}")
    
    # Print mutation matrix statistics
    Q = p['Q']
    print(f"\nMutation Matrix Statistics:")
    print(f"  Diagonal (no mutation): {np.diag(Q).mean():.6f} ± {np.diag(Q).std():.6f}")
    off_diag = Q[~np.eye(m, dtype=bool)]
    print(f"  Off-diagonal (mutation): {off_diag.mean():.6e} ± {off_diag.std():.6e}")
    print(f"  Max mutation rate: {off_diag.max():.6e}")
    print(f"  Min mutation rate: {off_diag[off_diag > 0].min():.6e}")
    
    # Define concentration sweep
    # Logarithmic spacing to capture dynamics at low and high concentrations
    C_min, C_max = 0.01, 6.0  # Extended to 6.0 to see full transition
    n_points = 30  # Increased for smoother curves
    C_values = np.logspace(np.log10(C_min), np.log10(C_max), n_points)
    
    print(f"\nRunning concentration sweep...")
    print(f"  {n_points} concentrations from {C_min} to {C_max}")
    print(f"  Simulation time: 100 time units per concentration")
    print(f"  (Extended range to capture full resistance transition)\n")
    
    # Run sweep
    results = antibiotic_concentration_sweep(
        p, 
        C_values, 
        uR=0.5,  # Resource influx
        T_final=100.0,
        x0_profile='susceptible_dominant',  # Start with susceptible-dominant
        verbose=True
    )
    
    # Print statistics
    print_summary_statistics(results, p)
    
    # Create visualizations
    print("Generating visualizations...")
    output_dir = Path(__file__).parent
    plot_sweep_results(results, p, output_dir=output_dir)
    
    # Save numerical results
    output_file = output_dir / f'resistance_sweep_results_{mutation_type}.npz'
    np.savez(output_file,
            C_values=results['C_values'],
            final_populations=results['final_populations'],
            total_biomass=results['total_biomass'],
            resistant_fraction=results['resistant_fraction'],
            diversity_index=results['diversity_index'],
            IC50=p['IC50'],
            rmax=p['rmax'],
            d0=p['d0'],
            Q=Q,
            mutation_type=mutation_type)
    print(f"Saved numerical results to: {output_file}")
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("  1. Check the heatmap for resistance emergence patterns")
    print("  2. Observe the critical antibiotic concentration")
    print("  3. Note diversity collapse at high concentrations")
    print("  4. Examine fitness costs of resistance (growth rates)")
    print(f"  5. Mutation model: {mutation_type}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()