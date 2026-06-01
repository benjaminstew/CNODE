from src.models.node_pyomo_base_admm import NODEPyomo
import numpy as np
import multiprocessing as mp
from scipy.integrate import solve_ivp

# ---- MODULE LEVEL FUNCTIONS (required for mp.Pool pickling) ----
WORKER = None # child processes (independent memory) each have own WORKER instance 

def init_worker(id_queue: mp.Queue, submodel_config: dict):
    global WORKER
    worker_id = id_queue.get() # = submodel index, per order of worker process creation
    WORKER = MPWorker(worker_id, submodel_config)

def solve_worker(payload: dict):
    WORKER.broadcast_consensus_and_duals(
        payload["consensus_Ws"],
        payload["consensus_bs"],
        payload["dual_var_Ws"][WORKER.worker_id],
        payload["dual_var_bs"][WORKER.worker_id]
    )
    WORKER.solve_submodel(payload["solver_options"])
    return (WORKER.worker_id, WORKER.submodel_Ws, WORKER.submodel_bs)


class MPWorker:
    '''
    Wraps a NODEPyomo submodel and handles all Pyomo <-> numpy parameter communication.
    Used directly in the main process for sequential mode, or inside worker processes for
    parallel mode.
    '''
    def __init__(self, worker_id: int, submodel_config: dict):
        self.worker_id = worker_id
        self.layer_sizes = submodel_config['layer_sizes']
        self.submodel = NODEPyomo(
            Y_obs=submodel_config['Y_obs'][worker_id],
            layer_sizes=self.layer_sizes,
            end_time=submodel_config['end_time'],
            state_lower_bound=submodel_config['state_lower_bound'],
            state_upper_bound=submodel_config['state_upper_bound'],
            param_lower_bound=submodel_config['param_lower_bound'],
            param_upper_bound=submodel_config['param_upper_bound'],
            l2_reg_param=submodel_config['l2_reg_param'],
            admm_submodel=True,
            admm_penalty_param=submodel_config['admm_penalty_param'],
            transcription_method=submodel_config['transcription_method'],
            residual_reg_param=submodel_config['residual_reg_param'],
            num_res_eval_nodes=submodel_config['num_res_eval_nodes'],
        )
        self.submodel_Ws = [0 for _ in range(len(self.layer_sizes) - 1)]
        self.submodel_bs = [0 for _ in range(len(self.layer_sizes) - 1)]

    def dict_to_ndarray(self, d: dict, shape: tuple):
        arr = np.zeros(shape, dtype=float)
        for (i, j), val in d.items():
            arr[i-1, j-1] = float(val)
        return arr

    def ndarray_to_dict(self, arr: np.ndarray):
        return {(i+1, j+1): float(val) for (i, j), val in np.ndenumerate(arr)}

    def solve_submodel(self, solver_options: dict):
        self.submodel.solve_model(solver_options)

        # extract submodel parameters after solve 
        for l in range(1, len(self.layer_sizes)):
            dict_Ws = self.submodel.model.Ws[l].W.extract_values()
            dict_bs = self.submodel.model.bs[l].b.extract_values()
            self.submodel_Ws[l-1] = self.dict_to_ndarray(dict_Ws, (self.layer_sizes[l-1], self.layer_sizes[l]))
            self.submodel_bs[l-1] = self.dict_to_ndarray(dict_bs, (1, self.layer_sizes[l]))

    def broadcast_consensus_and_duals(self,
            consensus_Ws: list, consensus_bs: list,
            dual_var_Ws: list, dual_var_bs: list
    ):  
        for l in range(1, len(self.layer_sizes)):
            self.submodel.model.consensus_Ws[l].W.store_values(self.ndarray_to_dict(consensus_Ws[l-1]))
            self.submodel.model.consensus_bs[l].b.store_values(self.ndarray_to_dict(consensus_bs[l-1]))
            self.submodel.model.dual_var_Ws[l].W.store_values(self.ndarray_to_dict(dual_var_Ws[l-1]))
            self.submodel.model.dual_var_bs[l].b.store_values(self.ndarray_to_dict(dual_var_bs[l-1]))

            # warm start submodel from consensus params (to help convergence)
            self.submodel.model.Ws[l].W.set_values(self.ndarray_to_dict(consensus_Ws[l-1]))
            self.submodel.model.bs[l].b.set_values(self.ndarray_to_dict(consensus_bs[l-1]))


class PoolManager:
    """
    Manages the multiprocessing pool lifecycle and parallel submodel execution.
    MPWorker instances live entirely inside the child processes; this class only
    opens/closes the pool and dispatches work via Pool.map.
    """
    def __init__(self, num_batches: int, submodel_config: dict):
        self.num_batches = num_batches
        self.pool = self.open_pool(submodel_config)

    def open_pool(self, submodel_config: dict):
        ctx = mp.get_context("spawn")
        id_queue = ctx.Queue()
        for i in range(self.num_batches):
            id_queue.put(i)
        pool = ctx.Pool(
            processes=self.num_batches,
            initializer=init_worker,
            initargs=(id_queue, submodel_config)
        )
        print(f"Opened multiprocessing pool with {self.num_batches} workers.")
        return pool

    def map(self, payload: dict) -> list:
        '''Dispatch payload to all workers; return list of (worker_id, submodel_Ws, submodel_bs).'''
        return self.pool.map(solve_worker, [payload] * self.num_batches, chunksize=1)

    def close(self):
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
            except Exception:
                self.pool.terminate()
                self.pool.join()
            finally:
                self.pool = None
            print("Closed multiprocessing pool.")


class ADMM:
    """
    Top-level interface for training Pyomo-based NODEs with ADMM and multiprocessing.
    Owns the consensus parameters, dual variables, and all ADMM algorithm logic.
    Set use_pool=False (default) to run sequentially for testing.
    """
    def __init__(self,
            Y_obs, layer_sizes, end_time,
            state_lower_bound, state_upper_bound,
            param_lower_bound, param_upper_bound,
            l2_reg_param,
            admm_penalty_param,
            max_admm_iterations,
            admm_convergence_tol,
            use_pool=False,
            transcription_method='dc',
            residual_reg_param=None,
            num_res_eval_nodes=None
    ):
        self.Y_obs = Y_obs # ndarray (num_batches, batch_size, state_dim)
        self.num_batches, self.batch_size, self.state_dim = self.Y_obs.shape
        self.layer_sizes = layer_sizes
        self.end_time = end_time
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.param_lower_bound = param_lower_bound
        self.param_upper_bound = param_upper_bound
        self.l2_reg_param = l2_reg_param
        self.admm_penalty_param = admm_penalty_param
        self.max_admm_iterations = max_admm_iterations
        self.admm_convergence_tol = admm_convergence_tol
        self.use_pool = use_pool
        self.transcription_method = transcription_method
        self.residual_reg_param = residual_reg_param
        self.num_res_eval_nodes = num_res_eval_nodes

        submodel_config = {
            'Y_obs': Y_obs,
            'layer_sizes': layer_sizes,
            'end_time': end_time,
            'state_lower_bound': state_lower_bound,
            'state_upper_bound': state_upper_bound,
            'param_lower_bound': param_lower_bound,
            'param_upper_bound': param_upper_bound,
            'l2_reg_param': l2_reg_param,
            'admm_penalty_param': admm_penalty_param,
            'transcription_method': transcription_method,
            'residual_reg_param': residual_reg_param,
            'num_res_eval_nodes': num_res_eval_nodes,
        }

        if use_pool:
            self.workers = None # MPWorker instances live in child processes
            self.pool_manager = PoolManager(self.num_batches, submodel_config)
        else:
            self.workers = [MPWorker(i, submodel_config) for i in range(self.num_batches)] 
            self.pool_manager = None

        self.dual_var_Ws, self.dual_var_bs = self._zero_initialise_dual_vars()
        self.submodels_Ws = [[0 for _ in range(len(layer_sizes) - 1)] for _ in range(self.num_batches)]
        self.submodels_bs = [[0 for _ in range(len(layer_sizes) - 1)] for _ in range(self.num_batches)]
        self.consensus_Ws = [np.zeros((layer_sizes[l], layer_sizes[l+1])) for l in range(len(layer_sizes) - 1)]
        self.consensus_bs = [np.zeros((1, layer_sizes[l+1])) for l in range(len(layer_sizes) - 1)]
        self.iteration_count = 0

    def _zero_initialise_dual_vars(self):
        dual_var_Ws = [
            [np.zeros((self.layer_sizes[l], self.layer_sizes[l+1])) for l in range(len(self.layer_sizes) - 1)]
            for _ in range(self.num_batches)
        ]
        dual_var_bs = [
            [np.zeros((1, self.layer_sizes[l+1])) for l in range(len(self.layer_sizes) - 1)]
            for _ in range(self.num_batches)
        ]
        return dual_var_Ws, dual_var_bs


    # ------------ SUBMODEL SOLVE METHODS -----------------

    def solve_submodels_sequential(self, solver_options: dict):
        for i in range(self.num_batches):
            self.workers[i].broadcast_consensus_and_duals(
                self.consensus_Ws, self.consensus_bs,
                self.dual_var_Ws[i], self.dual_var_bs[i]
            )
            self.workers[i].solve_submodel(solver_options)
            self.submodels_Ws[i] = self.workers[i].submodel_Ws
            self.submodels_bs[i] = self.workers[i].submodel_bs

    def solve_submodels_parallel(self, solver_options: dict):
        payload = {
            "consensus_Ws": self.consensus_Ws,
            "consensus_bs": self.consensus_bs,
            "dual_var_Ws": self.dual_var_Ws,
            "dual_var_bs": self.dual_var_bs,
            "solver_options": solver_options,
        }
        results = self.pool_manager.map(payload)
        for worker_id, submodel_Ws, submodel_bs in results:
            self.submodels_Ws[worker_id] = submodel_Ws
            self.submodels_bs[worker_id] = submodel_bs


    # --------------- ADMM UPDATE METHODS -------------------

    def update_consensus_params(self):
        '''Update consensus params by averaging submodel params.'''
        self.consensus_Ws = [w.copy() for w in self.submodels_Ws[0]]
        self.consensus_bs = [b.copy() for b in self.submodels_bs[0]]
        for i in range(1, self.num_batches):
            for l in range(len(self.layer_sizes) - 1):
                self.consensus_Ws[l] += self.submodels_Ws[i][l]
                self.consensus_bs[l] += self.submodels_bs[i][l]
        for l in range(len(self.layer_sizes) - 1):
            self.consensus_Ws[l] /= self.num_batches
            self.consensus_bs[l] /= self.num_batches

    def update_dual_vars(self):
        for i in range(self.num_batches):
            for l in range(len(self.layer_sizes) - 1):
                self.dual_var_Ws[i][l] += (
                    self.admm_penalty_param * (self.submodels_Ws[i][l] - self.consensus_Ws[l])
                )
                self.dual_var_bs[i][l] += (
                    self.admm_penalty_param * (self.submodels_bs[i][l] - self.consensus_bs[l])
                )

    def compute_primal_residual(self):
        r_primal = 0
        for i in range(self.num_batches):
            for l in range(len(self.layer_sizes) - 1):
                r_primal += np.linalg.norm(self.submodels_Ws[i][l] - self.consensus_Ws[l])
                r_primal += np.linalg.norm(self.submodels_bs[i][l] - self.consensus_bs[l])
        return r_primal

    #def compute_dual_residual(self):


    #----------------- MAIN ADMM LOOP -------------------

    def run_admm_training(self, solver_options: dict):
        near_tol_count = 0
        near_tol_upper = 1.5 * self.admm_convergence_tol

        try:
            for iteration in range(self.max_admm_iterations):
                print(f"ADMM iteration {iteration + 1}")
                if self.use_pool:
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
        finally:
            if self.use_pool:
                self.pool_manager.close()


    # ---------------- NODE PREDICTION METHODS --------------------

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
    