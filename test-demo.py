#!/usr/bin/env python3
"""
Live Demo Script for Bacterial Evolution Network Visualizer

Shows real-time animation - pops up in a window, no files saved.

Usage:
    python demo_live.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualize_network import live_evolution_demo


def create_demo_parameters(m=12):
    """Create parameters for demo with realistic non-uniform mutation matrix."""
    np.random.seed(42)
    
    IC50 = np.linspace(0.1, 5.0, m)
    rmax = 1.0 - 0.02 * (IC50 - IC50.min()) / (IC50.max() - IC50.min())
    d0 = 0.15 + 0.05 * (IC50 - IC50.min()) / (IC50.max() - IC50.min())
    K = 0.4 * np.ones(m)
    alpha = 0.1 * np.ones(m)
    h = 1.5 * np.ones(m)
    
    # Create biologically-realistic mutation matrix (fitness_landscape model)
    # Based on Drake 1991 mutation rates and Andersson & Hughes 2010 asymmetry
    Q = np.zeros((m, m))
    IC50_norm = (IC50 - IC50.min()) / (IC50.max() - IC50.min() + 1e-10)
    
    base_mutation_rate = 1e-8  # Per cell per generation (Drake 1991)
    population_size = 1e7      # Typical infection
    effective_rate = base_mutation_rate * population_size
    
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            
            # Distance-dependent mutation probability (exponential decay)
            distance = abs(IC50_norm[i] - IC50_norm[j])
            rate = effective_rate * np.exp(-5.0 * distance)
            
            # Strong asymmetry: 10x easier to LOSE resistance than gain it
            # (Andersson & Hughes 2010 - loss of function easier than gain)
            if IC50[j] < IC50[i]:  # Losing resistance
                rate *= 10.0
            
            Q[i, j] = min(rate, 0.01)
    
    # Normalize rows
    for i in range(m):
        row_sum = Q[i].sum()
        if row_sum > 0.05:
            Q[i] *= 0.05 / row_sum
        Q[i, i] = 1.0 - Q[i].sum()
    
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


def main():
    """Run live evolution demo."""
    print("\n" + "="*60)
    print("LIVE EVOLUTION ANIMATION")
    print("="*60)
    print("\nWatch as antibiotic concentration increases from 0.1 to 5.0")
    print("  - Blue nodes = susceptible (low IC50)")
    print("  - Red nodes = resistant (high IC50)")  
    print("  - Node SIZE = population density")
    print("  - Gold ring = dominant genotype")
    print("\nClose the window to exit...")
    print("="*60 + "\n")
    
    p = create_demo_parameters(m=12)
    
    # Run live demo - displays in real-time!
    anim = live_evolution_demo(p, C_range=(0.1, 5.0), n_frames=40, interval=250)
    
    return anim


if __name__ == '__main__':
    main()