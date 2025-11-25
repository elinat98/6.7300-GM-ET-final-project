"""
Forward Euler ODE integrator implementation.

Implements the explicit Forward Euler method:
    x[n+1] = x[n] + ∆t * f(x[n], p, u(t[n]))

For solving dx/dt = f(x, p, u(t))
"""

import numpy as np
import time
from typing import Callable, Tuple, Optional


def forward_euler(eval_f: Callable,
                  x0: np.ndarray,
                  p: dict,
                  eval_u: Callable,
                  t_start: float,
                  t_stop: float,
                  dt: float,
                  verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward Euler integrator with fixed time step.
    
    Parameters
    ----------
    eval_f : callable
        Function that evaluates f(x, p, u) where dx/dt = f(x, p, u)
        Signature: f = eval_f(x, p, u)
        x should be shape (N,) or (N,1), returns shape (N,) or (N,1)
    x0 : array
        Initial state vector, shape (N,)
    p : dict
        Parameters for the model
    eval_u : callable
        Function that evaluates input vector u(t)
        Signature: u = eval_u(t)
    t_start : float
        Starting time
    t_stop : float
        Stopping time
    dt : float
        Fixed time step size (∆t)
    verbose : bool
        If True, print progress information
    
    Returns
    -------
    X : array, shape (N, num_steps+1)
        State trajectory, X[:, n] is state at time t[n]
    t : array, shape (num_steps+1,)
        Time points
    """
    
    # Ensure x0 is 1D
    x0_flat = np.asarray(x0, dtype=float).ravel()
    N = x0_flat.size
    
    # Determine number of steps
    num_steps = int(np.ceil((t_stop - t_start) / dt))
    
    # Adjust dt to exactly reach t_stop
    dt_adjusted = (t_stop - t_start) / num_steps
    
    # Pre-allocate arrays
    X = np.zeros((N, num_steps + 1))
    t = np.zeros(num_steps + 1)
    
    # Set initial condition
    X[:, 0] = x0_flat
    t[0] = t_start
    
    if verbose:
        print(f"Forward Euler: {num_steps} steps with dt = {dt_adjusted:.6e}")
    
    # Integration loop
    for n in range(num_steps):
        t[n+1] = t[n] + dt_adjusted
        
        # Evaluate right-hand side at current state
        x_n = X[:, n]
        u_n = eval_u(t[n])
        f_n = eval_f(x_n, p, u_n)
        
        # Ensure f_n is 1D for consistent addition
        f_n_flat = np.asarray(f_n, dtype=float).ravel()
        
        # Forward Euler step: x[n+1] = x[n] + dt * f[n]
        X[:, n+1] = x_n + dt_adjusted * f_n_flat
        
        if verbose and (n + 1) % max(1, num_steps // 10) == 0:
            print(f"  Step {n+1}/{num_steps}, t = {t[n+1]:.4f}")
    
    # Ensure final time is exactly t_stop
    t[-1] = t_stop
    
    if verbose:
        print(f"Integration complete. Final state norm: {np.linalg.norm(X[:, -1]):.6e}")
    
    return X, t

