#!/usr/bin/env python3
"""
Network Visualizer for Bacterial Evolution Model

Visualizes the bacterial genotype network where:
- Nodes = bacterial genotypes
- Node SIZE = population density (larger = more bacteria)
- Node COLOR = resistance level (IC50): blue (susceptible) → red (resistant)
- Edge THICKNESS = mutation probability between genotypes
- Edge COLOR = mutation direction (gain vs loss of resistance)

This visualization is useful for:
1. Debugging Newton iterations (steady-state solving)
2. Understanding evolutionary dynamics
3. Identifying dominant genotypes and mutation pathways
4. Creating compelling demos for presentations

Usage:
    from visualize_network import visualize_network
    fig, ax = visualize_network(x, p, title="My State")
    plt.show()
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def visualize_network(x, p, ax=None, title=None, show_edges=True, edge_threshold=1e-6,
                      node_scale=200, edge_scale=3.0,
                      show_labels=True, show_colorbars=True, time=None,
                      antibiotic_conc=None, highlight_dominant=True):
    """
    Visualize the bacterial evolution network.
    
    Parameters
    ----------
    x : array-like
        State vector [n1, n2, ..., nm, R, C] where ni = population of genotype i
    p : dict
        Parameter dictionary containing:
        - 'Q': mutation matrix (m x m)
        - 'IC50': resistance levels for each genotype
        - 'rmax': max growth rates (optional, for tooltip)
        - 'd0': death rates (optional)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title
    show_edges : bool
        Whether to show mutation edges
    edge_threshold : float
        Minimum mutation probability to display edge
    node_scale : float
        Scaling factor for node sizes
    edge_scale : float
        Scaling factor for edge widths
    show_labels : bool
        Whether to show genotype labels on nodes
    show_colorbars : bool
        Whether to show color/size legends
    time : float, optional
        Current simulation time (for display)
    antibiotic_conc : float, optional
        Current antibiotic concentration (for display)
    highlight_dominant : bool
        Whether to highlight the dominant genotype
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    
    # Extract state
    x = np.asarray(x).flatten()
    m = len(p['IC50'])
    
    populations = x[:m]
    R = x[m] if len(x) > m else None
    C = x[m+1] if len(x) > m+1 else antibiotic_conc
    
    # Get parameters
    Q = p['Q']
    IC50 = np.asarray(p['IC50'])
    
    # Normalize IC50 to [0, 1] for coloring
    IC50_norm = (IC50 - IC50.min()) / (IC50.max() - IC50.min() + 1e-10)
    
    # Normalize populations for sizing (use log scale for better visualization)
    pop_safe = np.maximum(populations, 1e-10)
    pop_log = np.log10(pop_safe + 1)
    pop_norm = (pop_log - pop_log.min()) / (pop_log.max() - pop_log.min() + 1e-10)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Compute node positions using spring layout
    positions = _compute_layout(m, IC50)
    
    # Draw edges first (so nodes are on top)
    if show_edges:
        _draw_edges(ax, positions, Q, IC50, edge_threshold, edge_scale)
    
    # Draw nodes
    node_sizes = node_scale * (0.2 + 0.8 * pop_norm)  # Min size 20% of max
    
    # Color map: blue (susceptible) to red (resistant)
    cmap = plt.cm.coolwarm
    node_colors = cmap(IC50_norm)
    
    # Draw nodes as circles
    for i in range(m):
        circle = Circle(positions[i], radius=np.sqrt(node_sizes[i])/50,
                       facecolor=node_colors[i], edgecolor='black',
                       linewidth=1.5, alpha=0.85, zorder=10)
        ax.add_patch(circle)
        
        # Add label
        if show_labels:
            # Determine text color based on background
            text_color = 'white' if IC50_norm[i] > 0.6 else 'black'
            ax.text(positions[i, 0], positions[i, 1], f'G{i+1}',
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   color=text_color, zorder=11)
    
    # Highlight dominant genotype
    if highlight_dominant and populations.max() > 0:
        dom_idx = np.argmax(populations)
        highlight = Circle(positions[dom_idx], 
                          radius=np.sqrt(node_sizes[dom_idx])/50 + 0.05,
                          facecolor='none', edgecolor='gold',
                          linewidth=3, linestyle='-', zorder=9)
        ax.add_patch(highlight)
    
    # Set axis limits with generous padding to ensure all nodes fit
    max_node_radius = np.sqrt(node_sizes.max()) / 50
    padding = 0.4 + max_node_radius
    
    # Use larger symmetric limits for better spread
    max_lim = 1.6  # Increased from 1.4
    ax.set_xlim(-max_lim, max_lim)
    ax.set_ylim(-max_lim, max_lim)
    
    # Add colorbars and legends
    if show_colorbars:
        _add_legends(fig, ax, cmap, IC50, populations, node_scale)
    
    # Add title and info
    title_parts = []
    if title:
        title_parts.append(title)
    if time is not None:
        title_parts.append(f't = {time:.1f}')
    if C is not None:
        title_parts.append(f'C = {C:.2f}')
    if R is not None:
        title_parts.append(f'R = {R:.2f}')
    
    if title_parts:
        ax.set_title('  |  '.join(title_parts), fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics text box
    total_pop = populations.sum()
    resistant_frac = populations[IC50 > np.median(IC50)].sum() / (total_pop + 1e-10)
    n_surviving = np.sum(populations > 0.01 * total_pop)
    
    # Shannon diversity
    p_i = populations / (total_pop + 1e-10)
    p_i = p_i[p_i > 0]
    diversity = -np.sum(p_i * np.log(p_i + 1e-10))
    
    stats_text = (f'Total Pop: {total_pop:.2f}\n'
                  f'Resistant: {resistant_frac*100:.1f}%\n'
                  f'Surviving: {n_surviving}/{m}\n'
                  f'Diversity: {diversity:.2f}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    return fig, ax


def _compute_layout(m, IC50, layout='spring'):
    """Compute node positions using spring (force-directed) layout."""
    
    # Start with grid-like initialization for better spread
    np.random.seed(42)
    cols = int(np.ceil(np.sqrt(m)))
    positions = np.zeros((m, 2))
    
    # Initialize in loose grid with more space
    for i in range(m):
        row = i // cols
        col = i % cols
        positions[i] = [(col - cols/2) * 1.5, (row - cols/2) * 1.5]  # More spacing
    
    # Add noise
    positions += np.random.randn(m, 2) * 0.4
    
    # Spring simulation with strong repulsion
    for iteration in range(300):
        forces = np.zeros_like(positions)
        
        for i in range(m):
            for j in range(i+1, m):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff) + 1e-6
                
                # Strong repulsion to prevent overlap
                repulsion_strength = 1.5 / (dist ** 1.5)
                force = repulsion_strength * diff / dist
                forces[i] += force
                forces[j] -= force
                
                # Very weak attraction for similar IC50
                ic50_dist = abs(IC50[i] - IC50[j]) / (IC50.max() - IC50.min() + 1e-6)
                if ic50_dist < 0.15:
                    attraction = 0.008 * dist
                    forces[i] -= attraction * diff / dist
                    forces[j] += attraction * diff / dist
        
        # Apply forces with damping
        damping = 0.5 if iteration < 150 else 0.2
        positions += forces * damping
        
        # Center the layout
        positions -= positions.mean(axis=0)
    
    # Normalize to safe bounds
    max_extent = np.abs(positions).max()
    if max_extent > 0:
        positions = positions / max_extent * 1.1  # Use more space
    
    return positions


def _draw_edges(ax, positions, Q, IC50, threshold, scale):
    """Draw mutation edges between nodes."""
    m = len(IC50)
    
    # Find max mutation probability for scaling
    Q_no_diag = Q.copy()
    np.fill_diagonal(Q_no_diag, 0)
    max_prob = Q_no_diag.max()
    
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            
            prob = Q[i, j]
            if prob < threshold:
                continue
            
            # Edge width scales with mutation probability
            # Use log scale to make differences more visible
            if prob > 0:
                normalized_prob = prob / (max_prob + 1e-12)
                width = scale * (0.3 + 2.7 * normalized_prob)  # Range: 0.3 to 3.0 * scale
            else:
                width = scale * 0.1
            
            width = np.clip(width, 0.1, 5.0)
            
            # Color based on direction: gaining resistance (red) vs losing (blue)
            if IC50[j] > IC50[i]:
                color = 'firebrick'  # Gaining resistance
                alpha = 0.3 + 0.3 * normalized_prob  # More visible for higher prob
            else:
                color = 'steelblue'  # Losing resistance
                alpha = 0.4 + 0.3 * normalized_prob
            
            # Draw curved arrow
            start = positions[i]
            end = positions[j]
            
            # Shorten arrow to not overlap with nodes
            direction = end - start
            dist = np.linalg.norm(direction)
            if dist < 0.1:
                continue
            direction = direction / dist
            
            # Offset start and end
            node_radius = 0.08
            start_adj = start + direction * node_radius
            end_adj = end - direction * node_radius
            
            # Draw line with arrow
            ax.annotate('', xy=end_adj, xytext=start_adj,
                       arrowprops=dict(arrowstyle='->', color=color,
                                      lw=width, alpha=alpha,
                                      connectionstyle='arc3,rad=0.1'))


def _add_legends(fig, ax, cmap, IC50, populations, node_scale):
    """Add colorbars and size legend."""
    
    # Resistance colorbar (right side)
    divider = make_axes_locatable(ax)
    cax_color = divider.append_axes("right", size="3%", pad=0.1)
    
    norm = mcolors.Normalize(vmin=IC50.min(), vmax=IC50.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax_color)
    cbar.set_label('IC50 (Resistance)', fontsize=10)
    
    # Size legend (bottom)
    # Create fake scatter points for legend
    size_legend_ax = fig.add_axes([0.15, 0.02, 0.3, 0.03])
    size_legend_ax.axis('off')
    
    pop_range = [populations.min(), populations.mean(), populations.max()]
    pop_labels = ['Low', 'Med', 'High']
    
    for i, (pop, label) in enumerate(zip(pop_range, pop_labels)):
        pop_log = np.log10(max(pop, 1e-10) + 1)
        pop_max_log = np.log10(max(populations.max(), 1e-10) + 1)
        pop_norm = pop_log / (pop_max_log + 1e-10)
        size = node_scale * (0.2 + 0.8 * pop_norm)
        
        size_legend_ax.scatter([i * 0.4], [0], s=size, c='gray', alpha=0.7)
        size_legend_ax.text(i * 0.4, -0.5, f'{label}\n({pop:.1f})', 
                           ha='center', va='top', fontsize=8)
    
    size_legend_ax.set_xlim(-0.2, 1.0)
    size_legend_ax.set_ylim(-1.5, 0.5)
    size_legend_ax.set_title('Population Size', fontsize=9, pad=2)


def visualize_network_animation(X_history, p, times=None, C_values=None,
                                interval=200, save_path=None, show=True):
    """
    Create an animation of the network evolution over time.
    
    Parameters
    ----------
    X_history : array (n_steps, n_states)
        State history from simulation
    p : dict
        Parameters
    times : array, optional
        Time points for each state
    C_values : array, optional
        Antibiotic concentrations at each step
    interval : int
        Milliseconds between frames
    save_path : str, optional
        Path to save animation (e.g., 'evolution.gif'). If None, only displays.
    show : bool
        Whether to display the animation live (default True)
        
    Returns
    -------
    anim : FuncAnimation object
    """
    from matplotlib.animation import FuncAnimation
    
    n_steps = X_history.shape[0] if X_history.ndim > 1 else 1
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update(frame):
        ax.clear()
        x = X_history[frame] if X_history.ndim > 1 else X_history
        t = times[frame] if times is not None else frame
        C = C_values[frame] if C_values is not None else None
        
        visualize_network(x, p, ax=ax, time=t, antibiotic_conc=C,
                         show_colorbars=False, title='Evolution')
        return ax,
    
    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval, 
                        blit=False, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000//interval)
        print(f"Animation saved to: {save_path}")
    
    if show:
        plt.show()
    
    return anim


def live_evolution_demo(p, C_range=(0.1, 5.0), n_frames=30, interval=300):
    """
    Run a live demo showing evolution under increasing antibiotic pressure.
    
    This function displays the animation in real-time without saving.
    
    Parameters
    ----------
    p : dict
        Model parameters
    C_range : tuple
        (min, max) antibiotic concentration range
    n_frames : int
        Number of animation frames
    interval : int
        Milliseconds between frames
        
    Returns
    -------
    anim : FuncAnimation object
    """
    from matplotlib.animation import FuncAnimation
    
    m = len(p['IC50'])
    IC50 = p['IC50']
    
    # Generate evolution trajectory
    C_values = np.linspace(C_range[0], C_range[1], n_frames)
    times = np.linspace(0, 100, n_frames)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update(frame):
        ax.clear()
        
        C = C_values[frame]
        t = times[frame]
        
        # Create state with realistic population dynamics
        # At low C: susceptible genotypes dominate (more fit without antibiotic stress)
        # At high C: resistant genotypes take over (survive antibiotic pressure)
        x = np.zeros(m + 2)
        
        for j in range(m):
            # Hill inhibition - sharper transition (higher Hill coefficient)
            h_coeff = p.get('h', np.ones(m))[j] if hasattr(p.get('h', 1.5), '__len__') else 2.0
            inhibition = 1 / (1 + (C / IC50[j]) ** h_coeff)
            
            # Fitness cost for resistance - resistant strains pay a price without antibiotic
            # This makes them RARE at low antibiotic concentrations
            resistance_level = (IC50[j] - IC50.min()) / (IC50.max() - IC50.min())
            fitness_cost = 1.0 - 0.25 * resistance_level  # 25% fitness penalty for max resistance
            
            # At LOW C: fitness cost dominates (susceptible strains win)
            # At HIGH C: antibiotic inhibition dominates (resistant strains win)
            relative_fitness = inhibition * fitness_cost
            
            # Base population - starts larger for susceptible strains
            if C < 1.0:
                # Low antibiotic: susceptible strains are abundant, resistant are rare
                base_pop = 15.0 * relative_fitness * np.exp(-2.0 * resistance_level)
            else:
                # Medium to high antibiotic: selection shifts to resistant
                base_pop = 15.0 * relative_fitness
            
            # At high C, amplify the winner (positive feedback / competitive exclusion)
            if C > 2.5:
                # Strong selection for resistance
                selection_pressure = (C - 2.5) / (C_range[1] - 2.5)  # 0 to 1
                
                # Resistant strains grow explosively, susceptible collapse
                amplification = 1.0 + 5.0 * selection_pressure * (relative_fitness ** 3)
                base_pop *= amplification
            
            x[j] = max(base_pop, 0.001)  # Minimum detectable population
        
        # At very high C, kill off the losers dramatically
        if C > 3.5:
            threshold = 0.03 * x[:m].max()
            x[:m] = np.where(x[:m] < threshold, x[:m] * 0.05, x[:m])
        
        x[m] = 0.5  # Resources
        x[m + 1] = C  # Antibiotic
        
        visualize_network(x, p, ax=ax, time=t, antibiotic_conc=C,
                         show_colorbars=False, show_edges=True,
                         title=f'Live Evolution Demo')
        
        # Add frame counter
        ax.text(0.98, 0.02, f'Frame {frame+1}/{n_frames}', 
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax,
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval,
                        blit=False, repeat=True)
    
    plt.show()
    
    return anim


def create_comparison_visualization(states, p, labels, title='State Comparison'):
    """
    Visualize multiple states side by side for comparison.
    
    Parameters
    ----------
    states : list of arrays
        List of state vectors to compare
    p : dict
        Parameters
    labels : list of str
        Labels for each state
    title : str
        Overall title
    """
    n_states = len(states)
    
    fig, axes = plt.subplots(1, n_states, figsize=(5*n_states, 5))
    if n_states == 1:
        axes = [axes]
    
    for i, (x, label) in enumerate(zip(states, labels)):
        visualize_network(x, p, ax=axes[i], title=label,
                         show_colorbars=(i == n_states - 1),
                         show_edges=True)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


if __name__ == '__main__':
    # Run tests when executed directly
    print("Running visualize_network tests...")
    print("Use 'python test_visualizer.py' for comprehensive tests")