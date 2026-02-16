import numpy as np
from scipy.integrate import solve_ivp
from CNODE.src.direct_collocation import BarycentricInterpolation

class VanDerPolOscillator:
    """
    The forced Van der Pol oscillator - a 2D system represented by a
    second-order ODE, which can be split into two first-order ODEs. The ODE describes 
    a system with nonlinear damping of degree μ. 
    If μ = 0 the system is linear and undamped, and we have a simple harmonic oscillator.
    """
    def __init__(self, mu=1.0, external_force=1.0, angular_freq=1.0, u_0=0.0, v_0=1.0, time_invariant=False):
        self.mu = mu # nonlinearity/damping parameter
        self.external_force = external_force # external periodic force parameter
        self.angular_freq = angular_freq
        self.u_0 = u_0 # initial displacement
        self.v_0 = v_0 # initial velocity
        self.time_invariant = time_invariant

    def get_initial_state(self):
        '''
        Returns the initial state vector of shape (2, ) containing the
        initial displacement (u_0) and initial velocity (v_0).
        '''
        if self.time_invariant:
            return np.array([self.u_0, self.v_0], dtype=np.float32)
        else:
            return np.array([self.u_0, self.v_0, 0], dtype=np.float32)

    def get_dynamics(self, t, y):
        '''
        The sequential solver calls this method at each time step t to update system state vector y.
        '''
        u = y[0]
        v = y[1]
        dudt = v
        dvdt = self.mu * (1 - u**2) * v - u + self.external_force * np.cos(self.angular_freq * t)
        if self.time_invariant:
            return np.array([dudt, dvdt], dtype=np.float32)
        else:
            return np.array([dudt, dvdt, t], dtype=np.float32)
    
def create_vdpo_dataset(num_nodes, end_time, noise_sd, add_noise=True, time_invariant=False): 
    '''
    Computes single trajectory of VDPO state over [0, end_time] and splits into 50/50 train set and test set.
    
    Train grid: Chebyshev nodes on [0, end_time/2] (for collocation).
    Test grid:  Chebyshev nodes on [end_time/2, end_time] (for evaluation).
    
    The IVP is solved on the joint (sorted) grid so that train and test trajectories are consistent 
    (same continuous solution). The differentiation matrix D is computed on the train grid so it matches
    the collocation points exactly.
    '''
    if num_nodes % 2 != 0 or end_time % 2 != 0:
        raise ValueError("num_nodes and end_time must be even so train/test splits are the same size.")
    half_nodes = num_nodes // 2
    half_time = end_time // 2

    #1. Create the system 
    system = VanDerPolOscillator(time_invariant=time_invariant)

    #2. Build train and test grids 
    train_interp = BarycentricInterpolation(0, half_time, half_nodes)
    test_interp  = BarycentricInterpolation(half_time, end_time, half_nodes)
    train_grid = train_interp.collocation_grid # (half_nodes, )
    test_grid  = test_interp.collocation_grid # (half_nodes, )

    #3. Merge into a single strictly-increasing grid and solve the IVP once
    full_grid = np.unique(np.concatenate([train_grid, test_grid]))  # sorted + deduplicated
    y_0 = system.get_initial_state()
    solution = solve_ivp(
        fun=system.get_dynamics,
        t_span=(full_grid[0], full_grid[-1]),
        y0=y_0,
        t_eval=full_grid,
        method='RK45'
    )
    full_trajectory = solution.y.T  # (num_nodes, state_dim)
    smooth_trajectory = full_trajectory.copy()

    #4. Extract train/test rows by matching grid values
    train_idx = np.isin(full_grid, train_grid)
    test_idx  = np.isin(full_grid, test_grid)
    train_trajectory = full_trajectory[train_idx, :] # (half_nodes, state_dim)
    train_trajectory_smooth = smooth_trajectory[train_idx, :]
    test_trajectory = full_trajectory[test_idx, :] # (half_nodes, state_dim)

    #5. Add Gaussian noise to training data only
    if add_noise and noise_sd is not None and noise_sd > 0:
        noise = np.random.randn(*train_trajectory.shape).astype(np.float32) * float(noise_sd)
        train_trajectory += noise

    #6. Compute collocation-estimated differentiation matrix on the train grid 
    D = train_interp.compute_derivative_matrix() # (half_nodes, half_nodes)

    return full_grid, smooth_trajectory, train_grid, train_trajectory, train_trajectory_smooth, test_grid, test_trajectory, D