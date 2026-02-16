# currently lacks customisation on the numerical analysis methods used, ML architecture used, param initialisation methods, etc...
import warnings
import numpy as np 
from scipy.integrate import solve_ivp
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from .direct_collocation import BarycentricInterpolation

class NeuralODEPyomo:
    """
    Implementation of a base optimisation model to train collocation-based Neural ODEs using Pyomo and the IPOPT solver. 
    I have tried to make the code as scalable as possible with NN size etc., but there are still may improvements to be made.
    """
    def __init__(self, 
            Y_obs, D, layer_sizes,   
            state_lower_bound, state_upper_bound, 
            param_lower_bound, param_upper_bound,
            lambda_reg, 
            Y_smooth=None
    ):
        self.Y_obs = Y_obs # observed state trajectory from measurements. ndarray (num_nodes, state_dim)
        self.num_nodes, self.state_dim = self.Y_obs.shape
        if self.state_dim > self.num_nodes:
            warnings.warn("Y_obs should be structured such that each row represents a new collocation point.")
        self.D = D # collocation-estimated differentiation matrix. ndarray (num_nodes, num_nodes)
        self.layer_sizes = layer_sizes
        self.model = pyo.ConcreteModel()
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.param_lower_bound = param_lower_bound
        self.param_upper_bound = param_upper_bound
        self.lambda_reg = lambda_reg # regularisation hyperparameter for loss function 
        self.Y_smooth = Y_smooth if Y_smooth is not None else Y_obs # noise-free trajectory for Y_star init
        self.build_model()


    # ------------------------------ MODEL BUILDING METHODS ------------------------------- #

    def initialise_weight(self, shape):
        '''Xavier initialisation of a weight matrix as pyomo Var'''
        W = np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))
        def init_rule(model, i, j):
            return W[i-1, j-1] # W is a 0-indexed ndarray 
        return init_rule

    def initialise_bias(self, shape):
        '''Normal random initialisation of a bias vector as pyomo Var'''
        b = np.random.randn(*shape) * 0.1
        def init_rule(model, i, j):
            return b[0, i-1] # b is a 0-indexed ndarray 
        return init_rule
    
    def initialise_diff_matrix(self):
        '''Initialisation of collocation-estimated differentiation matrix as a pyomo Param.'''
        def init_rule(model, n, n_prime):
            return self.D[n-1, n_prime-1] # self.D is a 0-indexed ndarray
        return init_rule
    
    def initialise_Y_star(self):
        '''Initialisation of the approximate state trajectory Y_star as a pyomo Var using smooth data.'''
        def init_rule(model, n, d):
            return self.Y_smooth[n-1, d-1] # self.Y_smooth is a 0-indexed ndarray
        return init_rule
    
    def reg_mse_objective_rule(self, model):
        '''
        Returns the objective function - the regularised MSE
        between approximate (model.Y_star) and observed (self.Y_obs) trajectories - as a pyomo expression.
        Take model as input so can use as pyo.Objective rule.
        '''
        # MSE term: (1/N)||Y_star - Y_obs||_F^2
        mse = (1 / self.num_nodes) * sum(
            (model.Y_star[n, d] - self.Y_obs[n-1, d-1])**2
            for n in model.n 
            for d in model.d
        ) 
        # Regularised L2 norm term: lambda||theta||_2^2
        l2_norm = sum(
            model.Ws[l].W[i, j]**2
            for l in model.l
            for i in range(1, self.layer_sizes[l-1] + 1) 
            for j in range(1, self.layer_sizes[l] + 1)    
        ) + sum(
            model.bs[l].b[1, j]**2
            for l in model.l
            for j in range(1, self.layer_sizes[l] + 1)
        )
        reg_l2_norm = self.lambda_reg * l2_norm

        return mse + reg_l2_norm

    def build_model(self):
        # ------------ SETS ------------
        self.model.l = pyo.RangeSet(len(self.layer_sizes) - 1) # [1:len(self.layer_sizes)-1] as for n layers, have n-1 W matrices/b vectors
        self.model.n = pyo.RangeSet(self.num_nodes) 
        self.model.d = pyo.RangeSet(self.state_dim) 

        # ----- DECISION VARIABLES -----
        # Initialise approximate state trajectory
        self.model.Y_star = pyo.Var(
            self.model.n, 
            self.model.d, 
            domain=pyo.Reals, 
            initialize=self.initialise_Y_star(),  
            bounds=(self.state_lower_bound, self.state_upper_bound)
        )
        # Initialise NN parameters
        self.model.Ws = pyo.Block(self.model.l) 
        self.model.bs = pyo.Block(self.model.l) 
        for l in self.model.l: 
            self.model.Ws[l].W = pyo.Var(
                {i for i in range(1, self.layer_sizes[l-1] + 1)}, # self.layer_sizes is 0-indexed list 
                {j for j in range(1, self.layer_sizes[l] + 1)},
                domain=pyo.Reals,
                initialize=self.initialise_weight((self.layer_sizes[l-1], self.layer_sizes[l])),
                bounds=(self.param_lower_bound, self.param_upper_bound)
            )
            self.model.bs[l].b = pyo.Var(
                {1},
                {i for i in range(1, self.layer_sizes[l] + 1)},
                domain=pyo.Reals,
                initialize=self.initialise_bias((1, self.layer_sizes[l])),
                bounds=(self.param_lower_bound, self.param_upper_bound)
            )

        # -- INITIAL CONDITION CONSTRAINT --
        # Force initial state in Y_star to equal initial state in Y_obs
        self.model.initial_condition = pyo.ConstraintList()
        for d in self.model.d:
            self.model.initial_condition.add(self.model.Y_star[1, d] == self.Y_obs[0, d-1])
        
        # ----- COLLOCATION CONSTRAINT -----
        # Initialise the collocation-estimated differentiation matrix as a pyomo Param
        self.model.D = pyo.Param(
            self.model.n,  
            self.model.n, 
            domain=pyo.Reals, 
            initialize=self.initialise_diff_matrix()
        )
        # Define LHS of ODE (dY_star/dt) as matrix of pyomo expressions 
        self.model.ode_lhs = pyo.Expression( 
            self.model.n, 
            self.model.d, 
            rule=lambda model, n, d: sum(self.model.Y_star[j, d] * self.model.D[n, j] for j in model.n) 
        )
        # Compute output of NN as a ndarray of Pyomo expressions 
        nn_output = self.nn_feedforward_pyo() 
        # Define ODE constraint: dY_star/dt == NN(Y_star) 
        self.model.ode = pyo.ConstraintList()
        for n in range(2, self.num_nodes + 1): # skip n=1 (initial condition already enforced)
            for d in self.model.d:
                self.model.ode.add(self.model.ode_lhs[n, d] == nn_output[n-1, d-1])
   
        # ------------ OBJECTIVE -----------
        self.model.obj = pyo.Objective(rule=self.reg_mse_objective_rule, sense=pyo.minimize)


    # ----------------------------------- TRAINING METHODS ----------------------------------- #
    
    def nn_feedforward_pyo(self): 
        '''
        Compute a feedforward pass of the NN (act func = tanh, linear output layer).
        Uses self.model.Y_star as input (time invariant approximated states at collocation points).
        Returns: output layer activations as an ndarray (i, j) of pyo.Expressions (one for each element of Y-star)
        '''
        # Input activations = Y_star 
        activations = [self.model.Y_star]
        # For each layer
        for l in self.model.l:
            # Create ndarray to store expressions for current layer
            current_activation = np.empty((self.num_nodes, self.layer_sizes[l]), dtype=object) 
            # For each unit in layer
            for u in range(1, self.layer_sizes[l] + 1):
                # For each collocation point
                for i in self.model.n:
                    # Linear output: activations[l-1] . W + b
                    if l == 1: # if at first hidden layer (input activation is the 1-indexed self.model.Y_star)
                        linear_out = sum(activations[l-1][i, d] * self.model.Ws[l].W[d, u] for d in self.model.d) + self.model.bs[l].b[1, u]
                    else: # input activation is 0-indexed ndarray 
                        linear_out = sum(activations[l-1][i-1, k-1] * self.model.Ws[l].W[k, u] for k in range(1, self.layer_sizes[l-1] + 1)) + self.model.bs[l].b[1, u]
                    # Apply activation func: tanh for hidden layers, linear for output layer
                    if l < len(self.layer_sizes) - 1:
                        current_activation[i-1, u-1] = pyo.tanh(linear_out)
                    else:
                        current_activation[i-1, u-1] = linear_out
            activations.append(current_activation)

        return activations[-1]
    
    def freeze_vars(self):
        '''Fix values of vars in pyomo model'''
        self.model.Y_star.fix() 
        for l in self.model.l:
            self.model.Ws[l].W.fix()
            self.model.bs[l].b.fix()

    def solve_model(self):
        '''Solve the Pyomo optimisation model using IPOPT solver. Returns the solver result object.'''

        solver = pyo.SolverFactory("ipopt", executable="/opt/homebrew/bin/ipopt")
        print("Solver available?: {}".format(solver.available())) 
        print("\nInitial NN parameter summary stats (pre-IPOPT): ")
        self.check_param_values()

        solver.options['max_iter'] = 3000
        solver.options['tol'] = 1e-6 
        solver.options['acceptable_tol'] = 1e-4
        solver.options['acceptable_iter'] = 15 
        solver.options['nlp_scaling_method'] = 'gradient-based'
        solver.options['mu_strategy'] = 'adaptive'
        solver.options['print_level'] = 5

        result = solver.solve(self.model, tee=True)
        if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
            print("Solution is feasible and optimal")
            self.freeze_vars()
            print("NN parameters have been frozen")
            print("\nFinal NN parameter summary stats (post-IPOPT): ")
            self.check_param_values()
        else:
            print("Solution is not feasible and/or not optimal: {}".format(str(result.solver)))

        return result
    

    # ------------------------------ DIAGNOSTIC METHODS --------------------------------------- #

    def check_ode_residuals(self):
        '''Print and return residual error between NN prediction and ODE at collocation points.''' 

        Y_star = self.pyomo_var_to_numpy(self.model.Y_star, (self.num_nodes, self.state_dim))
        Ws = self.convert_weights()
        bs = self.convert_biases()
        
        # ODE LHS: D . Y_star 
        D_np = self.pyomo_var_to_numpy(self.model.D, (self.num_nodes, self.num_nodes))
        lhs = np.dot(D_np, Y_star) 
        # ODE RHS: NN(Y_star) 
        rhs = np.zeros_like(lhs)
        for n in range(self.num_nodes):
            rhs[n, :] = self.nn_prediction(0, Y_star[n, :], Ws, bs)
        # calc residual 
        residual = lhs - rhs

        # summary stats for all nodes 
        abs_res = np.abs(residual)
        all_max = abs_res.max()
        all_mean = abs_res.mean()

        # summary stats for constrained nodes (pyomo indices 2 -> N)
        constrained = abs_res[1:, :]
        cons_max = constrained.max() if constrained.size else float('nan')
        cons_mean = constrained.mean() if constrained.size else float('nan')

        # where is the worst constrained violation?
        worst_flat = int(np.argmax(constrained)) 
        worst_row, worst_dim = np.unravel_index(worst_flat, constrained.shape) 
        worst_n = worst_row + 2 # convert back to Pyomo indexing for reporting

        # print summary stats
        print(f"ODE residual (all nodes) max: {all_max:.6f}")
        print(f"ODE residual (all nodes) mean: {all_mean:.6f}")
        print(f"ODE residual (constrained nodes 2 -> N) max: {cons_max:.6f}")
        print(f"ODE residual (constrained nodes 2 -> N) mean: {cons_mean:.6f}")
        msg = f"Worst constrained node: n={worst_n}, dim={worst_dim+1}, |res|={constrained[worst_row, worst_dim]:.6f}"
        print(msg)

        return residual
    
    def check_param_values(self):
        '''Print summary statistics for NN parameters.'''
        for l in self.model.l:
            W = self.pyomo_var_to_numpy(self.model.Ws[l].W, (self.layer_sizes[l-1], self.layer_sizes[l]))
            b = self.pyomo_var_to_numpy(self.model.bs[l].b, (1, self.layer_sizes[l]))
            print(f"Layer {l}: W range [{W.min():.4f}, {W.max():.4f}], |W| mean {np.abs(W).mean():.4f}")
            print(f"Layer {l}: b range [{b.min():.4f}, {b.max():.4f}], |b| mean {np.abs(b).mean():.4f}")


    # ---------------------------- PREDICTION/IVP SOLVE METHODS -------------------------------- #

    def pyomo_var_to_numpy(self, pyo_obj, shape): 
        '''Copy pyo_obj (a Pyomo Var/Param) into a numpy ndarray.'''
        return np.array([[pyo.value(pyo_obj[i, j]) for j in range(1, shape[1]+1)] for i in range(1, shape[0]+1)])
    
    def convert_weights(self):
        return [self.pyomo_var_to_numpy(self.model.Ws[l].W, (self.layer_sizes[l-1], self.layer_sizes[l])) for l in self.model.l]
    
    def convert_biases(self):
        return [self.pyomo_var_to_numpy(self.model.bs[l].b, (1, self.layer_sizes[l])) for l in self.model.l]
    
    def nn_prediction(self, t, y_0, Ws, bs): 
        y_0 = y_0.reshape(1, -1)  # reshape to row vector
        out = np.tanh(np.dot(y_0, Ws[0]) + bs[0])  # first hidden layer
        for W, b in zip(Ws[1:-1], bs[1:-1]): 
            out = np.tanh(np.dot(out, W) + b)  # calc activation for hidden layer, move forward
        out = np.dot(out, Ws[-1]) + bs[-1]  # linear output layer
        
        return out.flatten()  # return 1D array for solve_ivp

    def get_predicted_trajectory(
        self, y_0, t_grid,
        method: str='RK45',
        rtol: float= 1e-3,
        atol: float= 1e-6,
        max_step: float= np.inf
    ):
        '''
        Predict state trajectory over t_grid via forward integration of the trained NN using scipy.solve_ivp.

        Args:
            y_0: Initial state, ndarray (state_dim,). Must be 1D for scipy.solve_ivp.
            t_grid: Strictly increasing time grid, ndarray (num_nodes,).
            method: solve_ivp method (e.g. 'RK45', 'DOP853', 'Radau', 'BDF').
            rtol, atol, max_step: solve_ivp tolerances/step control.

        Returns:
            Predicted state trajectory as ndarray (num_nodes, state_dim).
        '''
        if t_grid.ndim != 1 or t_grid.size < 2:
            raise ValueError(f"t_grid must be a 1D array with >=2 points; got shape {t_grid.shape}.")
        if not np.all(np.isfinite(t_grid)):
            raise ValueError("t_grid must contain only finite values.")
        if np.any(np.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing (solve_ivp expects monotone time).")

        if y_0.ndim != 1:
            raise ValueError(f"y_0 must be 1D; got shape {y_0.shape}.")
        if hasattr(self, 'state_dim') and y_0.shape[0] != self.state_dim:
            raise ValueError(f"y_0 has length {y_0.shape[0]} but state_dim is {self.state_dim}.")
        if not np.all(np.isfinite(y_0)):
            raise ValueError("y_0 must contain only finite values.")

        Ws = self.convert_weights()
        bs = self.convert_biases()

        predicted_trajectory = solve_ivp(
            fun=self.nn_prediction,
            t_span=(t_grid[0], t_grid[-1]),
            y0=y_0,
            t_eval=t_grid,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            args=(Ws, bs)
        )

        if not predicted_trajectory.success:
            raise RuntimeError("solve_ivp failed: " f"status={predicted_trajectory.status}, message={predicted_trajectory.message}")

        return predicted_trajectory.y.T

