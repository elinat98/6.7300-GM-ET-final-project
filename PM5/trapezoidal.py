import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add PS5_python_v1 to path if not already there
ps5_v1_path = os.path.join(os.path.dirname(__file__), 'PS5_python_v1')
if os.path.exists(ps5_v1_path) and ps5_v1_path not in sys.path:
    sys.path.insert(0, ps5_v1_path)

from visualize_state import visualize_state
from newtonNd import newtonNd

def trapezoidal(eval_f, eval_Jf, x_start, p, eval_u, t_start, t_stop, timestep, visualize=False):
    """
    Uses the Trapezoidal (implicit) algorithm to simulate the state model dx/dt = f(x, p, u)
    starting from state vector x_start at time t_start until time t_stop with time intervals of timestep.
    
    Since Trapezoidal is implicit, each time step requires solving a nonlinear system using Newton's method.
    
    Parameters:
    eval_f     - function to evaluate f(x, p, u)
    eval_Jf    - function to evaluate Jacobian J_f(x, p, u) of f
    x_start    - initial state vector
    p          - parameters needed for the function
    eval_u     - function to evaluate u(t)
    t_start    - start time
    t_stop     - stop time
    timestep   - time interval
    visualize  - if True, generates intermediate plots of the state
    
    Returns:
    X          - array of state vectors over time
    t          - array of time points
    """
    
    # Initialize arrays
    num_steps = int(np.ceil((t_stop - t_start) / timestep)) + 1
    X = np.zeros((len(x_start), num_steps))
    t = np.zeros(num_steps)
    
    # Set initial values
    X[:, 0] = x_start
    t[0] = t_start
    
    # Initialize visualization with two subplots
    if visualize:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
        fig.show()
    
    # Newton solver parameters for implicit solve at each time step
    # These are tight tolerances for accurate time integration
    errf = 1e-10        # Absolute equation error: how close f should be to zero
    errDeltax = 1e-10   # Absolute output error: how close x should be
    relDeltax = 1e-10   # Relative output error: percentage tolerance
    MaxIter = 50        # Maximum Newton iterations per time step
    FiniteDifference = 0  # Use analytical Jacobian (faster and more accurate)
    
    # Trapezoidal loop
    for n in range(num_steps - 1):
        dt = min(timestep, t_stop - t[n])
        t[n + 1] = t[n] + dt
        
        # Current state and inputs
        x_n = X[:, n]
        u_n = eval_u(t[n])
        u_np1 = eval_u(t[n + 1])
        
        # Evaluate f at current time step
        f_n = eval_f(x_n, p, u_n)
        
        # Define the nonlinear function to solve: Ftrap(x_{n+1}) = 0
        # Ftrap(x_{n+1}) = x_{n+1} - x_n - (dt/2) * [f(x_n, p, u_n) + f(x_{n+1}, p, u_{n+1})]
        def eval_Ftrap(x_new, p_trap, u_trap):
            """
            Evaluates Ftrap(x_new) = x_new - x_n - (dt/2) * [f(x_n, p, u_n) + f(x_new, p, u_{n+1})]
            
            Note: p_trap contains all necessary information (x_n, dt, f_n, eval_f, p, u_n, u_np1)
            """
            x_n = p_trap['x_n']
            dt = p_trap['dt']
            f_n = p_trap['f_n']
            eval_f = p_trap['eval_f']
            p_orig = p_trap['p']
            u_np1 = p_trap['u_np1']
            
            f_new = eval_f(x_new, p_orig, u_np1)
            Ftrap = x_new - x_n - (dt / 2.0) * (f_n + f_new)
            return Ftrap
        
        # Define the Jacobian of Ftrap
        # J_Ftrap = I - (dt/2) * J_f(x_new, p, u_{n+1})
        def eval_JFtrap(x_new, p_trap, u_trap):
            """
            Evaluates the Jacobian of Ftrap
            J_Ftrap = I - (dt/2) * J_f(x_new, p, u_{n+1})
            """
            dt = p_trap['dt']
            eval_Jf = p_trap['eval_Jf']
            p_orig = p_trap['p']
            u_np1 = p_trap['u_np1']
            
            N = len(x_new)
            I = np.eye(N)
            J_f = eval_Jf(x_new, p_orig, u_np1)
            J_Ftrap = I - (dt / 2.0) * J_f
            return J_Ftrap
        
        # Prepare parameters for Newton solver
        p_trap = {
            'x_n': x_n,
            'dt': dt,
            'f_n': f_n,
            'eval_f': eval_f,
            'eval_Jf': eval_Jf,
            'p': p,
            'u_np1': u_np1
        }
        
        # Initial guess for x_{n+1}: use Forward Euler prediction
        # This provides a good starting point for Newton iteration
        x_guess = x_n + dt * f_n
        
        # Solve the nonlinear system using Newton's method
        x_new, converged, errf_k, errDeltax_k, relDeltax_k, iterations, _ = newtonNd(
            eval_Ftrap, x_guess, p_trap, None, errf, errDeltax, relDeltax, 
            MaxIter, False, FiniteDifference, eval_JFtrap
        )
        
        if not converged:
            print(f'Warning: Newton did not converge at time step {n+1}, t = {t[n+1]:.6f}')
            print(f'  errf_k = {errf_k:.2e}, errDeltax_k = {errDeltax_k:.2e}, relDeltax_k = {relDeltax_k:.2e}')
        
        # Store the solution
        X[:, n + 1] = x_new
        
        # Update visualization
        if visualize:
            ax_top, ax_bottom = visualize_state(t[:n+2], X[:, :n+2], n + 1, '.b', ax_top, ax_bottom)
            plt.pause(0.001)
    
    if visualize:
        plt.show()
    
    return X, t

