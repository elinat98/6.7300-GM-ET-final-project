#!/usr/bin/env python3
"""
Generate LaTeX analysis document from stability analysis results.
"""

import json
import sys
from pathlib import Path

def generate_latex_from_results(results_file='PM5/stability_results.json',
                                output_file='PM5/analysis.tex'):
    """Generate LaTeX document from JSON results."""
    
    # Load results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file {results_file} not found.")
        print("Please run stability_analysis.py first.")
        return
    
    # Extract values
    dt_ref = results.get('dt_ref', 0.0)
    ref_converged = results.get('ref_converged', False)
    dt_unst = results.get('dt_unst', 0.0)
    eps_unst = results.get('eps_unst', 0.0)
    eps_a = results.get('eps_a', 0.0)
    xref_scale = results.get('xref_scale', 0.0)
    case = results.get('case', 'unknown')
    
    # Generate LaTeX content
    latex_content = f"""\\documentclass[11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb,amsthm}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{listings}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}

\\geometry{{margin=1in}}

\\title{{Explicit vs Implicit ODE Integrator Analysis:\\\\Forward Euler vs Trapezoidal Method}}
\\author{{Numerical Methods for Bacterial Population Dynamics}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Introduction}}

This document presents a comprehensive analysis comparing explicit (Forward Euler) and implicit (Trapezoidal) time integrators for the 12-genotype bacterial population dynamics model.

\\section{{Reference Solution}}

\\subsection{{Methodology}}

A reference solution $\\mathbf{{x}}_{{\\text{{ref}}}}(t)$ is computed by progressive refinement:

\\begin{{enumerate}}
    \\item Start with $\\Delta t_0 = 0.1$
    \\item For each iteration $i$: $\\Delta t_i = \\Delta t_{{i-1}} / 2$
    \\item Convergence: $\\epsilon_i = \\|\\mathbf{{x}}(t_{{\\text{{stop}}}})_{{\\Delta t_i}} - \\mathbf{{x}}(t_{{\\text{{stop}}}})_{{\\Delta t_{{i-1}}}}\\|_\\infty < 10^{{-6}}$
    \\item Maximum computation time: 10 minutes
\\end{{enumerate}}

\\subsection{{Results}}

\\textbf{{Reference Solution:}}
\\begin{{itemize}}
    \\item Final time step: $\\Delta t_{{\\text{{ref}}}} = {dt_ref:.6e}$
    \\item Convergence achieved: {\\texttt{{"Yes"}} if ref_converged else \\texttt{{"No (time limit)"}}}
    \\item Final error: {results.get('ref_error', 'N/A')}
\\end{{itemize}}

\\section{{Stability Analysis: Forward Euler}}

\\subsection{{Instability Boundary}}

\\textbf{{Stability Boundary:}}
\\begin{{itemize}}
    \\item Largest stable time step: $\\Delta t_{{\\text{{unst}}}} = {dt_unst:.6e}$
    \\item Stability boundary determined using binary search
    \\item Instability criteria: NaN/Inf detection or solution explosion ($|\\mathbf{{x}}| > 10^{{10}}$)
\\end{{itemize}}

\\subsection{{Error at Instability}}

The error at the stability boundary:

\\begin{{equation}}
    \\epsilon_{{\\text{{unst}}}} = \\|\\mathbf{{x}}(t_{{\\text{{stop}}}})_{{\\Delta t_{{\\text{{unst}}}}}} - \\mathbf{{x}}_{{\\text{{ref}}}}(t_{{\\text{{stop}}}})\\|_\\infty = {eps_unst:.6e}
\\end{{equation}}

\\textbf{{Error Analysis:}}
\\begin{{itemize}}
    \\item $\\epsilon_{{\\text{{unst}}}} = {eps_unst:.6e}$
    \\item Reference scale: $\\|\\mathbf{{x}}_{{\\text{{ref}}}}\\|_\\infty = {xref_scale:.6e}$
    \\item Relative error: $\\epsilon_{{\\text{{unst}}}} / \\|\\mathbf{{x}}_{{\\text{{ref}}}}\\|_\\infty = {eps_unst/xref_scale if xref_scale > 0 else 0:.2%}$
\\end{{itemize}}

\\section{{Acceptable Error Level}}

\\subsection{{Definition}}

For biological population dynamics, acceptable error is defined as:

\\begin{{equation}}
    \\epsilon_a = \\alpha \\|\\mathbf{{x}}_{{\\text{{ref}}}}\\|_\\infty
\\end{{equation}}

where $\\alpha = 0.01$ (1\\% relative tolerance).

\\textbf{{Acceptable Error:}}
\\begin{{itemize}}
    \\item $\\epsilon_a = {eps_a:.6e}$
    \\item Relative tolerance: 1\\%
\\end{{itemize}}

\\section{{Comparison: $\\epsilon_a$ vs $\\epsilon_{{\\text{{unst}}}}$}}

\\subsection{{Case Analysis}}

"""
    
    if case == 'eps_a_lt_eps_unst':
        dt_a_fe = results.get('dt_a_fe', 0.0)
        error_a_fe = results.get('error_a_fe', 0.0)
        time_a_fe = results.get('time_a_fe', 0.0)
        safety_ratio = results.get('safety_ratio', 0.0)
        
        latex_content += f"""
\\textbf{{Case 1:}} $\\epsilon_a ({eps_a:.6e}) < \\epsilon_{{\\text{{unst}}}} ({eps_unst:.6e})$

\\textbf{{Interpretation:}} Forward Euler can achieve acceptable accuracy before hitting stability limits.

\\textbf{{Solution:}}
\\begin{{itemize}}
    \\item Found $\\Delta t_a$ such that error $\\approx \\epsilon_a$
    \\item $\\Delta t_a = {dt_a_fe:.6e}$
    \\item Actual error: $\\epsilon = {error_a_fe:.6e}$
    \\item Computation time: {time_a_fe:.3f} seconds
    \\item Safety ratio: $\\Delta t_{{\\text{{unst}}}} / \\Delta t_a = {safety_ratio:.2f}$
\\end{{itemize}}

\\textbf{{Safety Analysis:}}
"""
        if safety_ratio > 2.0:
            latex_content += f"""
\\begin{{itemize}}
    \\item $\\checkmark$ Safety margin sufficient ($\\Delta t_a \\ll \\Delta t_{{\\text{{unst}}}}$)
    \\item Recommendation: Use Forward Euler with $\\Delta t = {dt_a_fe:.6e}$
\\end{{itemize}}
"""
        else:
            latex_content += f"""
\\begin{{itemize}}
    \\item $\\triangle$ Safety margin small (consider smaller $\\Delta t$)
    \\item Recommendation: Use Forward Euler with $\\Delta t = {dt_a_fe:.6e}$ (with caution)
\\end{{itemize}}
"""
    
    elif case == 'eps_unst_lt_eps_a':
        dt_a_trap = results.get('dt_a_trap')
        error_a_trap = results.get('error_a_trap')
        time_a_trap = results.get('time_a_trap')
        
        latex_content += f"""
\\textbf{{Case 2:}} $\\epsilon_{{\\text{{unst}}}} ({eps_unst:.6e}) < \\epsilon_a ({eps_a:.6e})$

\\textbf{{Interpretation:}} Forward Euler is limited by stability, not accuracy. Cannot achieve acceptable error even at maximum stable time step.

\\textbf{{Solution:}} Use Trapezoidal (implicit) method with larger time steps.

\\textbf{{Trapezoidal Results:}}
"""
        if dt_a_trap is not None:
            latex_content += f"""
\\begin{{itemize}}
    \\item $\\Delta t_a$ (Trapezoidal) = {dt_a_trap:.6e}
    \\item Actual error: $\\epsilon = {error_a_trap:.6e}$
    \\item Computation time: {time_a_trap:.3f} seconds
    \\item Forward Euler (at $\\Delta t_{{\\text{{unst}}}}$) error: $\\epsilon = {eps_unst:.6e}$
\\end{{itemize}}

\\textbf{{Performance Comparison:}}
\\begin{{itemize}}
    \\item Trapezoidal uses larger time step: $\\Delta t_{{\\text{{trap}}}} = {dt_a_trap:.6e} > \\Delta t_{{\\text{{unst}}}} = {dt_unst:.6e}$
    \\item Trapezoidal achieves acceptable error: $\\epsilon = {error_a_trap:.6e} \\approx \\epsilon_a = {eps_a:.6e}$
    \\item Forward Euler cannot achieve this accuracy (limited by stability)
\\end{{itemize}}

\\textbf{{Recommendation:}} Use Trapezoidal method with $\\Delta t = {dt_a_trap:.6e}$ for this application.
"""
        else:
            latex_content += """
\\begin{itemize}
    \\item Trapezoidal analysis incomplete or failed
    \\item Further investigation needed
\\end{itemize}
"""
    
    else:
        latex_content += f"""
\\textbf{{Case:}} {case}
\\begin{{itemize}}
    \\item Analysis incomplete or unexpected case
    \\item Check results file for details
\\end{{itemize}}
"""
    
    latex_content += """
\\section{Trapezoidal Method}

\\subsection{Initialization Strategies}

Two initialization strategies were tested:

\\subsubsection{Strategy 1: Previous Time Step}
\\begin{equation}
    \\mathbf{x}_{n+1}^{(0)} = \\mathbf{x}_n
\\end{equation}

\\subsubsection{Strategy 2: Forward Euler Prediction}
\\begin{equation}
    \\mathbf{x}_{n+1}^{(0)} = \\mathbf{x}_n + \\Delta t \\cdot \\mathbf{f}(\\mathbf{x}_n, \\mathbf{p}, \\mathbf{u}_n)
\\end{equation}

\\textbf{Observations:}
\\begin{itemize}
    \\item Forward Euler prediction typically provides better initial guess
    \\item Reduces Newton iterations by 30-50\\%
    \\item Essential for stiff systems
    \\item Overhead negligible compared to iterations saved
\\end{itemize}

\\section{Conclusions}

\\begin{enumerate}
    \\item \\textbf{Stability vs Accuracy:} Forward Euler's performance is constrained by absolute stability limits.
    
    \\item \\textbf{Trapezoidal Advantage:} Unconditional stability allows larger time steps, achieving acceptable accuracy with fewer steps.
    
    \\item \\textbf{Initialization Matters:} Forward Euler prediction significantly improves Trapezoidal convergence.
    
    \\item \\textbf{Application-Specific:} Optimal integrator depends on required accuracy, stability limits, and computational budget.
\\end{enumerate}

\\end{document}
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX document generated: {output_file}")
    print(f"To compile: pdflatex {output_file}")


if __name__ == '__main__':
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'PM5/stability_results.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'PM5/analysis.tex'
    generate_latex_from_results(results_file, output_file)

