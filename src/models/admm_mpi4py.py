'''DESCRIPTION OF FILE HERE'''
from src.models.node_pyomo_base_admm import NODEPyomo
import numpy as np
from scipy.integrate import solve_ivp
from mpi4py import MPI 

class ADMM():
    """
    Acts as top-level interface for training Pyomo-based NODEs with ADMM. 
    Builds and iteracts with the submodels, and implements the ADMM update steps and convergence check.
    Parallelisation of subproblem solves handled with mpi4py (multiple processes) within this class.
    """
    def __init__(self,
            Y_obs, layer_sizes, end_time, 
            state_lower_bound, state_upper_bound, 
            param_lower_bound, param_upper_bound,
            l2_reg_param, 
            admm_penalty_param, 
            max_admm_iterations,
            admm_convergence_tol,
            use_mpi=True, 
            transcription_method='dc',
            residual_reg_param=None,
            num_res_eval_nodes=None
    ):
        self.Y_obs = Y_obs # ndarray (num_batches, batch_size, state_dim)
        self.num_batches, self.batch_size, self.state_dim = self.Y_obs.shape
        
        # submodel hyperparams
        self.layer_sizes = layer_sizes
        self.end_time = end_time
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.param_lower_bound = param_lower_bound
        self.param_upper_bound = param_upper_bound
        self.l2_reg_param = l2_reg_param
        self.use_mpi = use_mpi
        self.transcription_method = transcription_method
        self.residual_reg_param = residual_reg_param
        self.num_res_eval_nodes = num_res_eval_nodes

        # admm hyperparams
        self.admm_penalty_param = admm_penalty_param
        self.max_admm_iterations = max_admm_iterations
        self.admm_convergence_tol = admm_convergence_tol
        
        # build submodels and initialise ADMM buffers
        self.submodels = self.build_submodels()
        self.dual_var_Ws, self.dual_var_bs = self.zero_initialise_dual_vars() # buffer for dual variable values - need to be zero initialised 
        self.submodels_Ws = [[0 for l in range(len(self.layer_sizes)-1)] for i in range(self.num_batches)] # buffers for submodel param values 
        self.submodels_bs = [[0 for l in range(len(self.layer_sizes)-1)] for i in range(self.num_batches)]
        self.consensus_Ws = [0 for l in range(len(self.layer_sizes)-1)] # buffers for consensus param values
        self.consensus_bs = [0 for l in range(len(self.layer_sizes)-1)]

        self.iteration_count = 0


    # ---------------------------- SUBMODEL BUILDING METHODS -------------------------------- #

    def dict_to_ndarray(self, dict, shape: tuple):
        arr = np.zeros(shape, dtype=float)
        for (i, j), val in dict.items():
            arr[i-1, j-1] = float(val)
        return arr

    def ndarray_to_dict(self, arr):
        return {(i+1, j+1): float(val) for (i, j), val in np.ndenumerate(arr)}

    def build_submodels(self):
        submodels = np.empty(self.num_batches, dtype=object)

        for i in range(self.num_batches):
            submodel = NODEPyomo(
                Y_obs=self.Y_obs[i], 
                layer_sizes=self.layer_sizes, 
                end_time=self.end_time, 
                state_lower_bound=self.state_lower_bound, 
                state_upper_bound=self.state_upper_bound, 
                param_lower_bound=self.param_lower_bound, 
                param_upper_bound=self.param_upper_bound, 
                l2_reg_param=self.l2_reg_param, 
                admm_submodel=True, 
                admm_penalty_param=self.admm_penalty_param, 
                transcription_method=self.transcription_method,
                residual_reg_param=self.residual_reg_param,
                num_res_eval_nodes=self.num_res_eval_nodes
            )
            submodels[i] = submodel
        
        return submodels
    
    def zero_initialise_dual_vars(self):
        # gather zero-initialised submodel dual var values - extract_values() returns dict of Pyomo Var values 
        dual_var_Ws = []
        dual_var_bs = []
        for i in range(self.num_batches):
            dual_var_Ws.append([])
            dual_var_bs.append([])
            for l in range(1, len(self.layer_sizes)):
                dict_Ws = self.submodels[i].model.dual_var_Ws[l].W.extract_values() 
                dict_bs = self.submodels[i].model.dual_var_bs[l].b.extract_values()
                dual_var_Ws[i].append(self.dict_to_ndarray(dict_Ws, (self.layer_sizes[l-1], self.layer_sizes[l]))) 
                dual_var_bs[i].append(self.dict_to_ndarray(dict_bs, (1, self.layer_sizes[l])))
        
        return dual_var_Ws, dual_var_bs
    

    # ------------------------------ ADMM UPDATE METHODS -------------------------------- #

    def solve_submodels_sequential(self, solver_options):
        for i in range(self.num_batches):
            # solve subproblem 
            self.submodels[i].solve_model(solver_options)

            # gather updated params from submodel
            for l in range(1, len(self.layer_sizes)):
                dict_Ws = self.submodels[i].model.Ws[l].W.extract_values()
                dict_bs = self.submodels[i].model.bs[l].b.extract_values()
                self.submodels_Ws[i][l-1] = self.dict_to_ndarray(
                    dict_Ws, (self.layer_sizes[l-1], self.layer_sizes[l])
                )
                self.submodels_bs[i][l-1] = self.dict_to_ndarray(
                    dict_bs, (1, self.layer_sizes[l])
                )

    def solve_submodels_parallel(self, solver_options):
        '''Solve submodels in parallel with MPI and gather parameter updates.'''
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()

        # if only one subproblem, skip MPI
        if size == 1:
            self.solve_submodels_sequential(solver_options)

        else:   
            local_results = []
            for i in range(rank, self.num_batches, size):
                # solve subproblem 
                self.submodels[i].solve_model(solver_options)
                Ws_layers = []
                bs_layers = []

                # gather updated params from submodel
                for l in range(1, len(self.layer_sizes)):
                    dict_Ws = self.submodels[i].model.Ws[l].W.extract_values()
                    dict_bs = self.submodels[i].model.bs[l].b.extract_values()
                    Ws_layers.append(
                        self.dict_to_ndarray(dict_Ws, (self.layer_sizes[l-1], self.layer_sizes[l]))
                    )
                    bs_layers.append(
                        self.dict_to_ndarray(dict_bs, (1, self.layer_sizes[l]))
                    )
                local_results.append((i, Ws_layers, bs_layers))

            gathered = MPI.COMM_WORLD.allgather(local_results)
            for rank_results in gathered:
                for i, Ws_layers, bs_layers in rank_results:
                    self.submodels_Ws[i] = Ws_layers
                    self.submodels_bs[i] = bs_layers

    def update_consensus_params(self): 
        '''update consensus params by averaging submodel params'''
        self.consensus_Ws = [w.copy() for w in self.submodels_Ws[0]]
        self.consensus_bs = [b.copy() for b in self.submodels_bs[0]]
        for i in range(1, self.num_batches):
            for l in range(len(self.layer_sizes)-1):
                self.consensus_Ws[l] += self.submodels_Ws[i][l] 
                self.consensus_bs[l] += self.submodels_bs[i][l]
        for l in range(len(self.layer_sizes)-1):
            self.consensus_Ws[l] /= self.num_batches
            self.consensus_bs[l] /= self.num_batches

        # broadcast consensus params back to submodels
        for i in range(self.num_batches):
            for l in range(1, len(self.layer_sizes)):
                dict_Ws = self.ndarray_to_dict(self.consensus_Ws[l-1]) 
                dict_bs = self.ndarray_to_dict(self.consensus_bs[l-1]) 
                self.submodels[i].model.consensus_Ws[l].W.store_values(dict_Ws)
                self.submodels[i].model.consensus_bs[l].b.store_values(dict_bs)

    def update_dual_vars(self):
        for i in range(self.num_batches):
            for l in range(1, len(self.layer_sizes)):
                # update dual variable for submodel
                self.dual_var_Ws[i][l-1] += (self.admm_penalty_param * (self.submodels_Ws[i][l-1] - self.consensus_Ws[l-1]))
                self.dual_var_bs[i][l-1] += (self.admm_penalty_param * (self.submodels_bs[i][l-1] - self.consensus_bs[l-1]))

                # broadcast updated dual variable back to submodel
                dict_Ws = self.ndarray_to_dict(self.dual_var_Ws[i][l-1])
                dict_bs = self.ndarray_to_dict(self.dual_var_bs[i][l-1])
                self.submodels[i].model.dual_var_Ws[l].W.store_values(dict_Ws)
                self.submodels[i].model.dual_var_bs[l].b.store_values(dict_bs)

    def compute_primal_residual(self):
        r_primal = 0
        for i in range(self.num_batches):
            for l in range(len(self.layer_sizes)-1):
                r_primal += np.linalg.norm(self.submodels_Ws[i][l] - self.consensus_Ws[l])
                r_primal += np.linalg.norm(self.submodels_bs[i][l] - self.consensus_bs[l])

        return r_primal
    
    def run_admm_training(self, solver_options):
        near_tol_count = 0
        near_tol_upper = 1.5 * self.admm_convergence_tol

        for iteration in range(self.max_admm_iterations):
            print(f"ADMM iteration {iteration+1}")
            if self.use_mpi:
                self.solve_submodels_parallel(solver_options)
            else:
                self.solve_submodels_sequential(solver_options)
            self.update_consensus_params()
            self.update_dual_vars()
            r_primal = self.compute_primal_residual()
            self.iteration_count += 1
            print(f"Primal residual: {r_primal:.6f}")
            if r_primal <= near_tol_upper:
                near_tol_count += 1
            if near_tol_count >= 3 or r_primal < self.admm_convergence_tol:
                print(f"ADMM converged after {self.iteration_count} iterations!")
                break

        if self.iteration_count >= self.max_admm_iterations:
            print("Max ADMM iterations reached.")


    # ---------------------------- PREDICTION METHODS -------------------------------- #

    def nn_prediction(self, t, y_0, Ws, bs):
        y_0 = y_0.reshape(1, -1)
        out = np.tanh(np.dot(y_0, Ws[0]) + bs[0])
        for W, b in zip(Ws[1:-1], bs[1:-1]):
            out = np.tanh(np.dot(out, W) + b)
        out = np.dot(out, Ws[-1]) + bs[-1]
        return out.flatten()

    def get_predicted_trajectory(self,
            y_0, t_grid,
            method: str = 'RK45',
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_step: float = np.inf
    ):
        '''
        Predict state trajectory over t_grid via forward integration of the trained NN.
        Uses consensus parameters directly — call after run_admm_training completes.
        '''
        predicted_trajectory = solve_ivp(
            fun=self.nn_prediction,
            t_span=(t_grid[0], t_grid[-1]),
            y0=y_0,
            t_eval=t_grid,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            args=(self.consensus_Ws, self.consensus_bs)
        )
        if not predicted_trajectory.success:
            raise RuntimeError(
                f"solve_ivp failed: status={predicted_trajectory.status}, message={predicted_trajectory.message}"
            )
        return predicted_trajectory.y.T







    
