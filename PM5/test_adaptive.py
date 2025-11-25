#!/usr/bin/env python3
"""
Test adaptive vs fixed time-stepping for Trapezoidal method.
"""

import sys
from pathlib import Path
import numpy as np
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evalf_bacterial import evalf
from jacobian_tools import evaljacobianf
from PM5.model_setup import setup_12_genotype_model
from PM5.adaptive_trapezoidal import compare_fixed_vs_adaptive

def main():
    """Run adaptive vs fixed comparison."""
    print("="*70)
    print("ADAPTIVE vs FIXED TIME-STEPPING COMPARISON")
    print("="*70)
    
    # Set up model
    print("\n[1/4] Setting up 12-genotype model...")
    p, x0, eval_u = setup_12_genotype_model()
    print(f"  Model initialized: {len(x0)} states")
    
    # Load reference solution
    print("\n[2/4] Loading reference solution...")
    try:
        with open('PM5/stability_results.json', 'r') as f:
            ref_data = json.load(f)
        xref = np.array(ref_data['xref'])
        error_target = ref_data['eps_a']
        t_start = 0.0
        t_stop = 5.0  # Adjust based on your simulation
        
        print(f"  Reference solution loaded")
        print(f"  Error target (ϵa): {error_target:.6e}")
        print(f"  Time range: [{t_start}, {t_stop}]")
    except FileNotFoundError:
        print("  Error: stability_results.json not found")
        print("  Run stability_analysis.py first")
        return
    
    # Run comparison
    print("\n[3/4] Running fixed vs adaptive comparison...")
    results = compare_fixed_vs_adaptive(
        eval_f=evalf,
        eval_Jf=evaljacobianf,
        x0=x0,
        p=p,
        eval_u=eval_u,
        t_start=t_start,
        t_stop=t_stop,
        xref=xref,
        error_target=error_target,
        dt_fixed=None,  # Will find automatically
        verbose=True
    )
    
    if results is None:
        print("  Comparison failed")
        return
    
    # Save results
    print("\n[4/4] Saving results...")
    output_file = 'PM5/adaptive_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved to {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Fixed Trapezoidal:")
    print(f"  Time: {results['fixed']['time']:.4f} s")
    print(f"  Steps: {results['fixed']['steps']}")
    print(f"  Error: {results['fixed']['error']:.6e}")
    print(f"\nAdaptive Trapezoidal:")
    print(f"  Time: {results['adaptive']['time']:.4f} s")
    print(f"  Steps: {results['adaptive']['steps']}")
    print(f"  Rejections: {results['adaptive']['rejections']}")
    print(f"  Error: {results['adaptive']['error']:.6e}")
    print(f"\nSpeedup: {results['speedup']:.2f}x")
    
    return results

if __name__ == '__main__':
    results = main()

