#!/usr/bin/env python3
"""
Create comprehensive plots for PM5 analysis.

Plots to create:
1. Reference solution convergence (error vs dt_ref)
2. Forward Euler error vs dt (showing dt_a and dt_unst)
3. Computation time vs dt
4. Solution trajectories comparison (different dt values)
5. State evolution over time (key variables)
6. Error comparison (bar chart)
7. Efficiency comparison (error vs time trade-off)
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
from PM5.forward_euler import forward_euler
from PM5.reference_solution import compute_reference_solution
from PM5.model_setup import setup_12_genotype_model


def plot_reference_convergence(ref_result, save_path=None):
    """Plot 1: Reference solution convergence."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Try to load convergence history from output file
    try:
        with open('PM5/analysis_output.txt', 'r') as f:
            lines = f.readlines()
        
        # Parse convergence history from output
        dt_values = []
        errors = []
        for line in lines:
            if 'Error: ||x_∆ti - x_∆ti-1||∞ =' in line:
                try:
                    error_str = line.split('=')[1].strip()
                    error = float(error_str)
                    # Find corresponding dt from previous line
                    prev_line_idx = lines.index(line) - 1
                    if prev_line_idx >= 0 and 'dt =' in lines[prev_line_idx]:
                        dt_str = lines[prev_line_idx].split('dt =')[1].split()[0]
                        dt = float(dt_str)
                        dt_values.append(dt)
                        errors.append(error)
                except:
                    continue
        
        if len(dt_values) > 1:
            ax.loglog(dt_values[:-1], errors[1:], 'o-', linewidth=2, markersize=8,
                     label='Error between successive dt values')
            
            # Add reference line for second-order convergence
            if len(dt_values) > 2:
                dt_fit = np.array(dt_values[1:-1])
                error_fit = np.array(errors[1:])
                # Fit line: log(error) = a + b*log(dt)
                p_fit = np.polyfit(np.log(dt_fit), np.log(error_fit), 1)
                dt_line = np.logspace(np.log10(dt_fit.min()), np.log10(dt_fit.max()), 100)
                error_line = np.exp(p_fit[1]) * dt_line**p_fit[0]
                ax.loglog(dt_line, error_line, '--', alpha=0.5, 
                         label=f'Fit: error ∝ dt^{p_fit[0]:.2f}')
            
            # Mark convergence point
            if len(errors) > 0 and errors[-1] < 1e-6:
                ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.7, 
                          label='Convergence tolerance')
                ax.plot(dt_values[-1], errors[-1], 'r*', markersize=15,
                       label=f'Converged: dt = {dt_values[-1]:.2e}')
    except Exception as e:
        print(f"  Warning: Could not load convergence history: {e}")
        ax.text(0.5, 0.5, 'Convergence history not available', 
               transform=ax.transAxes, ha='center', va='center')
    
    ax.set_xlabel('Time Step $\\Delta t$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error $||x_{\\Delta t_i} - x_{\\Delta t_{i-1}}||_\\infty$', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Reference Solution Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_error_vs_dt(p, x0, eval_u, t_start, t_stop, xref, dt_a, dt_unst, eps_a, 
                     eps_unst, dt_ref, save_path=None):
    """Plot 2: Forward Euler error vs dt (showing dt_a, dt_unst, dt_ref)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Generate dt values for testing
    dt_values = np.logspace(-3, 0, 20)  # 0.001 to 1.0
    errors = []
    times = []
    stable = []
    
    print("Computing error vs dt plot...")
    for i, dt in enumerate(dt_values):
        if i % 5 == 0:
            print(f"  {i}/{len(dt_values)}: dt = {dt:.6e}")
        
        try:
            start_time = time.time()
            X, t = forward_euler(evalf, x0, p, eval_u, t_start, t_stop, dt, verbose=False)
            comp_time = time.time() - start_time
            
            if X is None:
                errors.append(np.nan)
                times.append(np.nan)
                stable.append(False)
                continue
            
            # Check stability
            max_abs = np.max(np.abs(X))
            is_stable = (max_abs < 1e10 and not np.any(np.isnan(X)) and not np.any(np.isinf(X)))
            stable.append(is_stable)
            
            if is_stable:
                error = np.linalg.norm(X[:, -1] - xref, ord=np.inf)
                errors.append(error)
                times.append(comp_time)
            else:
                errors.append(np.nan)
                times.append(np.nan)
                
        except Exception as e:
            errors.append(np.nan)
            times.append(np.nan)
            stable.append(False)
    
    errors = np.array(errors)
    times = np.array(times)
    
    # Plot 1: Error vs dt (log-log)
    stable_mask = np.array(stable)
    unstable_mask = ~stable_mask
    
    ax1.loglog(dt_values[stable_mask], errors[stable_mask], 'o-', 
              color='blue', linewidth=2, markersize=6, label='Forward Euler (stable)')
    ax1.loglog(dt_values[unstable_mask], np.full(np.sum(unstable_mask), errors[stable_mask].max()*10),
              'x', color='red', markersize=8, label='Forward Euler (unstable)')
    
    # Mark key points
    ax1.axvline(x=dt_ref, color='green', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Reference: $\\Delta t_{{ref}} = {dt_ref:.2e}$')
    ax1.axvline(x=dt_a, color='orange', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Optimal: $\\Delta t_a = {dt_a:.4f}$')
    ax1.axvline(x=dt_unst, color='red', linestyle=':', alpha=0.7, linewidth=2,
               label=f'Stability limit: $\\Delta t_{{unst}} = {dt_unst:.2e}$')
    ax1.axhline(y=eps_a, color='purple', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Acceptable error: $\\epsilon_a = {eps_a:.4f}$')
    ax1.axhline(y=eps_unst, color='darkred', linestyle=':', alpha=0.7, linewidth=2,
               label=f'Error at instability: $\\epsilon_{{unst}} = {eps_unst:.2f}$')
    
    ax1.set_xlabel('Time Step $\\Delta t$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error $||x(\\Delta t) - x_{{ref}}||_\\infty$', 
                  fontsize=12, fontweight='bold')
    ax1.set_title('Forward Euler: Error vs Time Step', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    
    # Plot 2: Computation time vs dt
    ax2.loglog(dt_values[stable_mask], times[stable_mask], 's-',
              color='blue', linewidth=2, markersize=6, label='Computation time')
    
    # Mark dt_a
    dt_a_time = times[stable_mask][np.argmin(np.abs(dt_values[stable_mask] - dt_a))]
    ax2.plot(dt_a, dt_a_time, 'o', color='orange', markersize=12, 
            label=f'$\\Delta t_a$: {dt_a_time:.4f} s')
    
    ax2.set_xlabel('Time Step $\\Delta t$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Computation Time vs Time Step', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_solution_trajectories(p, x0, eval_u, t_start, t_stop, xref, 
                               dt_values=[0.5, 0.2, 0.1385, 0.05, 0.01],
                               save_path=None):
    """Plot 3: Solution trajectories for different dt values."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    m = 12  # Number of genotypes
    
    for dt in dt_values:
        try:
            X, t = forward_euler(evalf, x0, p, eval_u, t_start, t_stop, dt, verbose=False)
            if X is None:
                continue
            
            error = np.linalg.norm(X[:, -1] - xref, ord=np.inf)
            label = f'$\\Delta t = {dt:.4f}$ (error = {error:.4f})'
            
            # Plot 1: Total biomass (sum of populations)
            total_biomass = np.sum(X[:m, :], axis=0)
            axes[0].plot(t, total_biomass, '-', linewidth=2, label=label, alpha=0.8)
            
            # Plot 2: Resource R
            axes[1].plot(t, X[m, :], '-', linewidth=2, label=label, alpha=0.8)
            
            # Plot 3: Antibiotic C
            axes[2].plot(t, X[m+1, :], '-', linewidth=2, label=label, alpha=0.8)
            
        except Exception as e:
            print(f"  Failed for dt = {dt}: {e}")
    
    # Add reference solution
    X_ref, t_ref = forward_euler(evalf, x0, p, eval_u, t_start, t_stop, 
                                 dt_values[-1]*0.1, verbose=False)  # Very fine
    if X_ref is not None:
        axes[0].plot(t_ref, np.sum(X_ref[:m, :], axis=0), 'k--', linewidth=2,
                    label='Reference (fine dt)', alpha=0.5)
        axes[1].plot(t_ref, X_ref[m, :], 'k--', linewidth=2,
                    label='Reference (fine dt)', alpha=0.5)
        axes[2].plot(t_ref, X_ref[m+1, :], 'k--', linewidth=2,
                    label='Reference (fine dt)', alpha=0.5)
    
    axes[0].set_ylabel('Total Biomass $\\sum_i n_i$', fontsize=11, fontweight='bold')
    axes[0].set_title('Solution Trajectories: Different Time Steps', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Resource $R$', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time $t$', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Antibiotic $C$', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=9, loc='best')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_state_evolution(p, x0, eval_u, t_start, t_stop, dt_a, xref, save_path=None):
    """Plot 4: Detailed state evolution with optimal dt."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Run simulation with optimal dt
    X, t = forward_euler(evalf, x0, p, eval_u, t_start, t_stop, dt_a, verbose=False)
    
    m = 12
    
    # Plot 1: All genotype populations (top-left)
    for i in range(min(m, 12)):
        axes[0, 0].plot(t, X[i, :], '-', linewidth=1.5, alpha=0.7, 
                       label=f'Genotype {i+1}' if i < 6 else '')
    axes[0, 0].set_ylabel('Population $n_i$', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'Genotype Populations ($\\Delta t = {dt_a:.4f}$)', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Resource and Antibiotic (top-right)
    axes[0, 1].plot(t, X[m, :], '-', linewidth=2, label='Resource $R$', color='green')
    axes[0, 1].plot(t, X[m+1, :], '-', linewidth=2, label='Antibiotic $C$', color='red')
    axes[0, 1].set_ylabel('Concentration', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Resource and Antibiotic', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Genotype diversity (bottom-left)
    # Compute Shannon entropy
    n_genotypes = X[:m, :]
    total = np.sum(n_genotypes, axis=0)
    fractions = n_genotypes / (total + 1e-10)
    shannon = -np.sum(fractions * np.log(fractions + 1e-10), axis=0)
    
    axes[1, 0].plot(t, shannon, '-', linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Shannon Entropy', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Population Diversity', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Top 5 genotypes by final population (bottom-middle)
    final_pops = X[:m, -1]
    top_indices = np.argsort(final_pops)[-5:][::-1]
    for idx in top_indices:
        axes[1, 1].plot(t, X[idx, :], '-', linewidth=2, 
                       label=f'Genotype {idx+1} (final: {final_pops[idx]:.2f})')
    axes[1, 1].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Population', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Top 5 Genotypes', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Error evolution (bottom-left)
    # Compute error at each time point (if we have reference trajectory)
    # For now, just show final error
    final_error = np.linalg.norm(X[:, -1] - xref, ord=np.inf)
    axes[2, 0].bar(['Final Error'], [final_error], color='orange', alpha=0.7)
    axes[2, 0].axhline(y=0.066, color='red', linestyle='--', linewidth=2,
                      label='Acceptable $\\epsilon_a = 0.066$')
    axes[2, 0].set_ylabel('Error $||x - x_{ref}||_\\infty$', 
                         fontsize=11, fontweight='bold')
    axes[2, 0].set_title('Error at Final Time', fontsize=12, fontweight='bold')
    axes[2, 0].legend(fontsize=10)
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Total biomass evolution (bottom-right)
    total_biomass = np.sum(X[:m, :], axis=0)
    axes[2, 1].plot(t, total_biomass, '-', linewidth=2, color='darkblue')
    axes[2, 1].set_xlabel('Time $t$', fontsize=11, fontweight='bold')
    axes[2, 1].set_ylabel('Total Biomass', fontsize=11, fontweight='bold')
    axes[2, 1].set_title('Total Biomass Evolution', fontsize=12, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_comparison_summary(results, save_path=None):
    """Plot 5: Summary comparison (bar charts)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    dt_ref = results.get('dt_ref', 0)
    dt_a = results.get('dt_a_fe', 0)
    dt_unst = results.get('dt_unst', 0)
    eps_a = results.get('eps_a', 0)
    eps_unst = results.get('eps_unst', 0)
    error_a_fe = results.get('error_a_fe', 0)
    time_a_fe = results.get('time_a_fe', 0)
    
    # Plot 1: Time steps comparison
    methods = ['Reference\n$\\Delta t_{ref}$', 'Optimal\n$\\Delta t_a$', 
              'Stability\n$\\Delta t_{unst}$']
    dt_values = [dt_ref, dt_a, dt_unst]
    colors = ['green', 'orange', 'red']
    
    axes[0, 0].bar(methods, dt_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel('Time Step $\\Delta t$', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Time Step Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    for i, (method, dt, color) in enumerate(zip(methods, dt_values, colors)):
        axes[0, 0].text(i, dt, f'{dt:.2e}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    # Plot 2: Error comparison
    error_methods = ['Optimal\n$\\epsilon(\\Delta t_a)$', 'Acceptable\n$\\epsilon_a$',
                    'Instability\n$\\epsilon_{unst}$']
    error_values = [error_a_fe, eps_a, eps_unst]
    error_colors = ['orange', 'purple', 'darkred']
    
    axes[0, 1].bar(error_methods, error_values, color=error_colors, alpha=0.7,
                  edgecolor='black', linewidth=2)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel('Error $||x - x_{ref}||_\\infty$', 
                         fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Error Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    for i, (method, err) in enumerate(zip(error_methods, error_values)):
        axes[0, 1].text(i, err, f'{err:.4f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    
    # Plot 3: Computation time (if available)
    if time_a_fe > 0:
        axes[1, 0].bar(['Forward Euler\n$\\Delta t_a$'], [time_a_fe],
                      color='blue', alpha=0.7, edgecolor='black', linewidth=2)
        axes[1, 0].set_ylabel('Computation Time (seconds)', 
                             fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Computation Time', fontsize=12, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].text(0, time_a_fe, f'{time_a_fe:.4f} s', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    # Plot 4: Safety ratio
    if dt_a > 0:
        safety_ratio = dt_unst / dt_a
        axes[1, 1].bar(['Safety Ratio\n$\\Delta t_{unst} / \\Delta t_a$'], 
                      [safety_ratio], color='green', alpha=0.7,
                      edgecolor='black', linewidth=2)
        axes[1, 1].set_ylabel('Ratio', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Safety Margin', fontsize=12, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].text(0, safety_ratio, f'{safety_ratio:.2e}', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_efficiency_tradeoff(p, x0, eval_u, t_start, t_stop, xref, save_path=None):
    """Plot 6: Efficiency trade-off (error vs computation time)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Test multiple dt values
    dt_values = np.logspace(-2, 0, 15)  # 0.01 to 1.0
    errors = []
    times = []
    
    print("Computing efficiency trade-off...")
    for i, dt in enumerate(dt_values):
        if i % 3 == 0:
            print(f"  {i}/{len(dt_values)}: dt = {dt:.6e}")
        
        try:
            start_time = time.time()
            X, t = forward_euler(evalf, x0, p, eval_u, t_start, t_stop, dt, verbose=False)
            comp_time = time.time() - start_time
            
            if X is None:
                continue
            
            max_abs = np.max(np.abs(X))
            if max_abs < 1e10 and not np.any(np.isnan(X)):
                error = np.linalg.norm(X[:, -1] - xref, ord=np.inf)
                errors.append(error)
                times.append(comp_time)
            
        except Exception:
            continue
    
    if len(errors) > 0:
        errors = np.array(errors)
        times = np.array(times)
        
        # Scatter plot
        scatter = ax.scatter(times, errors, c=dt_values[:len(errors)], 
                           s=100, alpha=0.7, cmap='viridis', edgecolors='black')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time Step $\\Delta t$', fontsize=11, fontweight='bold')
        
        # Mark optimal point
        optimal_idx = np.argmin(np.abs(errors - 0.066))  # Closest to eps_a
        if len(errors) > optimal_idx:
            ax.plot(times[optimal_idx], errors[optimal_idx], 'r*', 
                   markersize=20, label=f'Optimal (near $\\epsilon_a$)')
        
        # Add reference lines
        ax.axhline(y=0.066, color='purple', linestyle='--', linewidth=2,
                  label='Acceptable error $\\epsilon_a = 0.066$')
        
        ax.set_xlabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error $||x - x_{ref}||_\\infty$', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency Trade-off: Error vs Computation Time', 
                    fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def main():
    """Generate all plots."""
    print("="*70)
    print("GENERATING PM5 ANALYSIS PLOTS")
    print("="*70)
    
    # Load results
    print("\n[1/6] Loading results...")
    with open('PM5/stability_results.json', 'r') as f:
        results = json.load(f)
    
    xref = np.array(results['xref'])
    dt_ref = results['dt_ref']
    dt_a = results['dt_a_fe']
    dt_unst = results['dt_unst']
    eps_a = results['eps_a']
    eps_unst = results['eps_unst']
    
    # Set up model
    print("\n[2/6] Setting up model...")
    p, x0, eval_u = setup_12_genotype_model()
    t_start, t_stop = 0.0, 5.0
    
    # Load reference convergence history if available
    ref_result = None  # Could load from file if saved
    
    # Generate plots
    print("\n[3/6] Plot 1: Reference convergence...")
    plot_reference_convergence(ref_result, save_path='PM5/plot1_reference_convergence.png')
    
    print("\n[4/6] Plot 2: Error vs dt...")
    plot_error_vs_dt(p, x0, eval_u, t_start, t_stop, xref, dt_a, dt_unst, eps_a,
                    eps_unst, dt_ref, save_path='PM5/plot2_error_vs_dt.png')
    
    print("\n[5/6] Plot 3: Solution trajectories...")
    plot_solution_trajectories(p, x0, eval_u, t_start, t_stop, xref,
                              save_path='PM5/plot3_solution_trajectories.png')
    
    print("\n[6/6] Plot 4: State evolution...")
    plot_state_evolution(p, x0, eval_u, t_start, t_stop, dt_a, xref,
                        save_path='PM5/plot4_state_evolution.png')
    
    print("\n[7/6] Plot 5: Comparison summary...")
    plot_comparison_summary(results, save_path='PM5/plot5_comparison_summary.png')
    
    print("\n[8/6] Plot 6: Efficiency trade-off...")
    plot_efficiency_tradeoff(p, x0, eval_u, t_start, t_stop, xref,
                            save_path='PM5/plot6_efficiency_tradeoff.png')
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED")
    print("="*70)
    print("\nPlots saved to PM5/ directory:")
    print("  - plot1_reference_convergence.png")
    print("  - plot2_error_vs_dt.png")
    print("  - plot3_solution_trajectories.png")
    print("  - plot4_state_evolution.png")
    print("  - plot5_comparison_summary.png")
    print("  - plot6_efficiency_tradeoff.png")


if __name__ == '__main__':
    main()

