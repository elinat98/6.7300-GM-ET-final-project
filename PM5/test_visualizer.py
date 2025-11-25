#!/usr/bin/env python3
"""
Test Script for Bacterial Evolution Network Visualizer

This script demonstrates the visualize_network function with various test cases:
1. Initial state (susceptible-dominant)
2. High antibiotic pressure (resistant-dominant)  
3. Mixed population (transitional state)
4. Concentration sweep comparison
5. Animation of evolution

Run this script to generate all test visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import visualizer
from visualize_network import (
    visualize_network,
    create_comparison_visualization,
    visualize_network_animation,
    live_evolution_demo
)


def create_test_parameters(m=12, mutation_type='fitness_landscape'):
    """Create test parameters for the bacterial evolution model."""
    np.random.seed(42)
    
    IC50 = np.linspace(0.1, 5.0, m)
    rmax = 1.0 - 0.02 * (IC50 - IC50.min()) / (IC50.max() - IC50.min())
    d0 = 0.15 + 0.05 * (IC50 - IC50.min()) / (IC50.max() - IC50.min())
    K = 0.4 * np.ones(m)
    alpha = 0.1 * np.ones(m)
    h = 1.5 * np.ones(m)
    
    Q = _create_mutation_matrix(m, IC50, mutation_type)
    
    return {
        'Q': Q,
        'IC50': IC50,
        'rmax': rmax,
        'd0': d0,
        'K': K,
        'alpha': alpha,
        'h': h,
        'kC': 0.05
    }


def _create_mutation_matrix(m, IC50, mutation_type='fitness_landscape'):
    """Create mutation matrix based on type."""
    Q = np.eye(m) * 0.95
    IC50_norm = (IC50 - IC50.min()) / (IC50.max() - IC50.min() + 1e-10)
    
    base_rate = 1e-8
    pop_size = 1e7
    effective_rate = base_rate * pop_size
    
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            distance = abs(IC50_norm[i] - IC50_norm[j])
            rate = effective_rate * np.exp(-5.0 * distance)
            
            if IC50[j] < IC50[i]:
                rate *= 10.0
            
            Q[i, j] = min(rate, 0.01)
    
    for i in range(m):
        Q[i, i] = 0
        row_sum = Q[i].sum()
        if row_sum > 0.05:
            Q[i] *= 0.05 / row_sum
        Q[i, i] = 1.0 - Q[i].sum()
    
    return Q


def create_test_state(p, scenario='susceptible_dominant'):
    """Create test state vectors for different scenarios."""
    m = len(p['IC50'])
    IC50 = p['IC50']
    
    x = np.zeros(m + 2)
    
    if scenario == 'susceptible_dominant':
        x[:m] = 10.0 * np.exp(-2.0 * (IC50 - IC50.min()) / (IC50.max() - IC50.min()))
        x[m] = 0.8
        x[m+1] = 0.1
        
    elif scenario == 'resistant_dominant':
        x[:m] = 10.0 * np.exp(-2.0 * (IC50.max() - IC50) / (IC50.max() - IC50.min()))
        x[m] = 0.3
        x[m+1] = 3.0
        
    elif scenario == 'mixed':
        mid = IC50.mean()
        x[:m] = 5.0 * np.exp(-1.0 * (np.abs(IC50 - mid)) / (IC50.max() - IC50.min()))
        x[m] = 0.5
        x[m+1] = 1.0
        
    elif scenario == 'extinct':
        x[:m] = 0.01
        x[np.argmax(IC50)] = 5.0
        x[m] = 0.9
        x[m+1] = 5.0
    
    return x


def test_basic_visualization():
    """Test 1: Basic visualization with different scenarios."""
    print("\n" + "="*60)
    print("TEST 1: Basic Visualization - Different Population Scenarios")
    print("="*60)
    
    p = create_test_parameters(m=12)
    
    scenarios = ['susceptible_dominant', 'mixed', 'resistant_dominant', 'extinct']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        x = create_test_state(p, scenario)
        C = x[-1]
        
        visualize_network(x, p, ax=axes[i], 
                         title=scenario.replace('_', ' ').title(),
                         antibiotic_conc=C,
                         show_colorbars=(i == 3))
    
    plt.suptitle('Test 1: Population Scenarios', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'test_basic_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def test_concentration_sweep():
    """Test 2: Concentration sweep comparison."""
    print("\n" + "="*60)
    print("TEST 2: Antibiotic Concentration Sweep")
    print("="*60)
    
    p = create_test_parameters(m=12)
    m_geno = len(p['IC50'])
    IC50 = p['IC50']
    
    C_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()
    
    for i, C in enumerate(C_values):
        x = np.zeros(m_geno + 2)
        
        for j in range(m_geno):
            inhibition = 1 / (1 + (C / IC50[j]) ** 1.5)
            fitness_cost = 1.0 - 0.1 * (IC50[j] - IC50.min()) / (IC50.max() - IC50.min())
            x[j] = 10.0 * inhibition * fitness_cost
        
        x[m_geno] = 0.5
        x[m_geno + 1] = C
        
        visualize_network(x, p, ax=axes[i],
                         title=f'C = {C:.1f}',
                         antibiotic_conc=C,
                         show_colorbars=(i == 5),
                         show_edges=False)
    
    plt.suptitle('Test 2: Effect of Antibiotic Concentration on Population',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'test_concentration_sweep.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def test_state_comparison():
    """Test 3: Side-by-side state comparison."""
    print("\n" + "="*60)
    print("TEST 3: State Comparison")
    print("="*60)
    
    p = create_test_parameters(m=10)
    
    states = [
        create_test_state(p, 'susceptible_dominant'),
        create_test_state(p, 'mixed'),
        create_test_state(p, 'resistant_dominant')
    ]
    labels = ['Before Treatment\n(C = 0.1)', 
              'During Treatment\n(C = 1.0)', 
              'After Treatment\n(C = 3.0)']
    
    fig = create_comparison_visualization(states, p, labels,
                                         title='Evolution Under Antibiotic Pressure')
    
    output_path = Path(__file__).parent / 'test_state_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def test_edge_visualization():
    """Test 4: Edge (mutation pathway) visualization."""
    print("\n" + "="*60)
    print("TEST 4: Mutation Pathway Visualization")
    print("="*60)
    
    p = create_test_parameters(m=8, mutation_type='fitness_landscape')
    x = create_test_state(p, 'mixed')
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    thresholds = [1e-4, 1e-5, 1e-6]
    
    for i, thresh in enumerate(thresholds):
        visualize_network(x, p, ax=axes[i],
                         title=f'Edge Threshold: {thresh:.0e}',
                         edge_threshold=thresh,
                         edge_scale=2.0,
                         show_colorbars=(i == 2))
    
    plt.suptitle('Test 4: Mutation Pathway Visibility (Edge Threshold)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'test_edge_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def test_animation():
    """Test 5: Animation of evolution."""
    print("\n" + "="*60)
    print("TEST 5: Evolution Animation")
    print("="*60)
    
    p = create_test_parameters(m=10)
    m_geno = len(p['IC50'])
    IC50 = p['IC50']
    
    n_frames = 20
    X_history = np.zeros((n_frames, m_geno + 2))
    times = np.linspace(0, 100, n_frames)
    C_values = np.linspace(0.1, 4.0, n_frames)
    
    for i, (t, C) in enumerate(zip(times, C_values)):
        for j in range(m_geno):
            inhibition = 1 / (1 + (C / IC50[j]) ** 1.5)
            fitness_cost = 1.0 - 0.1 * (IC50[j] - IC50.min()) / (IC50.max() - IC50.min())
            X_history[i, j] = 10.0 * inhibition * fitness_cost
        X_history[i, m_geno] = 0.5
        X_history[i, m_geno + 1] = C
    
    try:
        output_path = Path(__file__).parent / 'test_evolution_animation.gif'
        anim = visualize_network_animation(X_history, p, times, C_values,
                                          interval=300, 
                                          save_path=str(output_path),
                                          show=False)
        print(f"Saved: {output_path}")
        print("\nTo view LIVE animation, run:")
        print("  python demo_live.py")
    except Exception as e:
        print(f"Animation failed: {e}")


def run_all_tests():
    """Run all visualization tests."""
    print("\n" + "="*70)
    print("BACTERIAL EVOLUTION NETWORK VISUALIZER - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Visualization", test_basic_visualization),
        ("Concentration Sweep", test_concentration_sweep),
        ("State Comparison", test_state_comparison),
        ("Edge Visualization", test_edge_visualization),
        ("Animation", test_animation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASSED"))
            plt.close('all')
        except Exception as e:
            results.append((name, f"FAILED: {e}"))
            plt.close('all')
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        status_str = "✓" if status == "PASSED" else "✗"
        print(f"  {status_str} {name}: {status}")
    
    n_passed = sum(1 for _, s in results if s == "PASSED")
    print(f"\nTotal: {n_passed}/{len(results)} tests passed")
    print("="*70)
    
    print("\nGenerated files:")
    output_dir = Path(__file__).parent
    for f in sorted(output_dir.glob('test_*.png')):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob('test_*.gif')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    run_all_tests()