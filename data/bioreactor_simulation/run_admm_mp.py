"""
Run ADMM training for the synthetic bioreactor dataset with multiprocessing.
Run from repo root with 'python -m data.bioreactor_simulation.run_admm_mp'
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.admm_mp import ADMM

def main():
    # import training data
    data_path = "./data/bioreactor_simulation/noise_level_01_eps0.01.csv"
    state_cols = ['G', 'O', 'X', 'Xd', 'P', 'L', 'CO2']
    state_dim = len(state_cols)

    layer_sizes = [state_dim, 32, state_dim]
    print(f"Using layer sizes: {layer_sizes}")

    # split train and test batch ids
    all_batch_ids = np.arange(0, 50)
    train_batch_ids = np.random.choice(all_batch_ids, size=5, replace=False).tolist()
    remaining_batch_ids = np.setdiff1d(all_batch_ids, train_batch_ids)
    test_batch_ids = sorted(np.random.choice(remaining_batch_ids, size=3, replace=False).tolist())

    df = pd.read_csv(data_path)
    train_batches = {
        batch_id: df[df['Batch'] == f"Batch_{batch_id}"].reset_index(drop=True)
        for batch_id in train_batch_ids
    }
    test_batches = {
        batch_id: df[df['Batch'] == f"Batch_{batch_id}"].reset_index(drop=True)
        for batch_id in test_batch_ids
    }
    train_grid = train_batches[train_batch_ids[0]]['Time'].values
    Y_obs = np.array(
        [train_batches[batch_id][state_cols].values for batch_id in train_batch_ids]
    )
    test_grid = test_batches[test_batch_ids[0]]['Time'].values
    Y_test_obs = np.array(
        [test_batches[batch_id][state_cols].values for batch_id in test_batch_ids]
    )
    end_time = train_grid[-1]
    #print(f"Time grid shape: {train_grid.shape}")
    #print(f"Y_obs shape: {Y_obs.shape}")
    #print(f"Test grid shape: {test_grid.shape}")
    #print(f"Y_test_obs shape: {Y_test_obs.shape}")

    # run neural ode training with ADMM
    ipopt_options = {
        "max_iter": 500,
        "print_level": 5,
        "nlp_scaling_method": "gradient-based",
        "mu_strategy": "adaptive",
        "tol": 1e-6,
        "acceptable_tol": 1e-5,
        "acceptable_iter": 10,
    }

    # define and run ADMM training for DC 
    admm = ADMM(
        Y_obs=Y_obs,
        layer_sizes=layer_sizes,
        end_time=end_time,
        state_lower_bound=-5,
        state_upper_bound=50,
        param_lower_bound=-100,
        param_upper_bound=100,
        l2_reg_param=1e-4,
        admm_penalty_param=1, 
        max_admm_iterations=500,
        admm_primal_residual_tol=1,
        admm_train_mse_tol=None,
        use_pool=True,
        transcription_method='dc'
    )

    # record train/test MSE of the consensus model for the wall-clock convergence plots 
    dt = train_grid[1] - train_grid[0]
    dt_test = test_grid[1] - test_grid[0]
    mse_eval_options = {
        "train_grid": train_grid,
        "Y_obs": Y_obs,
        "test_grid": test_grid,
        "Y_test_obs": Y_test_obs,
        "rtol": 1e-7,
        "atol": 1e-9,
        "max_step": dt
    }

    t0 = time.perf_counter()
    admm.run_admm_training(ipopt_options, mse_eval_options=mse_eval_options)

    # define and run ADMM training for IRR-DC
    '''admm = ADMM(
        Y_obs=Y_obs,
        layer_sizes=layer_sizes,
        end_time=end_time,
        state_lower_bound=-5,
        state_upper_bound=50,
        param_lower_bound=-100,
        param_upper_bound=100,
        l2_reg_param=1e-4,
        admm_penalty_param=10,
        max_admm_iterations=500,
        admm_convergence_tol=0.1,
        use_pool=True,
        transcription_method='irrdc',
        residual_reg_param=0.5,
        num_res_eval_nodes=25
    )
    t0 = time.perf_counter()
    admm.run_admm_training(ipopt_options, mse_eval_options=mse_eval_options)'''

    print(f"ADMM training time: {time.perf_counter() - t0:.1f}s")

    # ---- wall-clock convergence plots (train/test MSE vs training time) ----
    history = admm.mse_history
    times = np.asarray(history["time_min"])

    for split, color in (("train", "tab:blue"), ("test", "tab:red")):
        mse = np.asarray(history[f"{split}_mse"])
        std = np.asarray(history[f"{split}_mse_std"])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(times, mse, color=color, marker='o', markersize=3, linewidth=1.4, label='Mean MSE')
        # translucent ±1 std band (clipped at a small positive floor for the log axis)
        lower = np.clip(mse - std, 1e-12, None)
        ax.fill_between(times, lower, mse + std, color=color, alpha=0.2,
                        label='±1 std (across batches)')
        ax.set_yscale('log')
        ax.set_xlabel('Training time (min)')
        ax.set_ylabel(f'{split.capitalize()}-set MSE')
        ax.grid(True, which='both', linestyle='--', color='lightgrey', alpha=0.8)
        ax.legend()
        plt.tight_layout()
        out = Path(__file__).with_name(f"{admm.transcription_method}_run1_{split}_convergence.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.show()

    # plot trajectories
    dt = train_grid[1] - train_grid[0]
    predicted_train = {}
    for batch_id in train_batch_ids:
        y0 = Y_obs[train_batch_ids.index(batch_id), 0, :]
        predicted_train[batch_id] = admm.get_predicted_trajectory(
            y0, train_grid, rtol=1e-7, atol=1e-9, max_step=dt
        )
    predicted_test = {}
    dt_test = test_grid[1] - test_grid[0]
    for batch_id in test_batch_ids:
        y0 = Y_test_obs[test_batch_ids.index(batch_id), 0, :]
        predicted_test[batch_id] = admm.get_predicted_trajectory(
            y0, test_grid, rtol=1e-7, atol=1e-9, max_step=dt_test
        )
    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(train_batch_ids)))

    for i, col in enumerate(state_cols):
        ax = axes[i]
        for c, batch_id in zip(colors, train_batch_ids):
            batch_idx = train_batch_ids.index(batch_id)
            obs = Y_obs[batch_idx]
            pred = predicted_train[batch_id]
            ax.scatter(
                train_grid, obs[:, i],
                color=c, marker='o', s=22, alpha=0.8,
                label=f'Observed Batch_{batch_id}', zorder=5
            )
            ax.plot(
                train_grid, pred[:, i],
                color=c, linewidth=1.6, linestyle='-',
                label=f'Predicted Batch_{batch_id}'
            )
        ax.set_title(col, fontsize=13)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration (g/L)')
        ax.grid(True, linestyle='--', color='lightgrey', alpha=0.8)

    axes[7].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc='lower right', bbox_to_anchor=(0.98, 0.02))
    plt.suptitle(f'Predicted vs Observed Trajectories (Training Batches; {admm.transcription_method})', fontsize=14)
    plt.tight_layout()
    output_path = Path(__file__).with_name(f"{admm.transcription_method}_run1_train.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_batch_ids)))

    for i, col in enumerate(state_cols):
        ax = axes[i]
        for c, batch_id in zip(colors, test_batch_ids):
            batch_idx = test_batch_ids.index(batch_id)
            obs = Y_test_obs[batch_idx]
            pred = predicted_test[batch_id]
            ax.scatter(
                test_grid, obs[:, i],
                color=c, marker='o', s=22, alpha=0.8,
                label=f'Observed Batch_{batch_id}', zorder=5
            )
            ax.plot(
                test_grid, pred[:, i],
                color=c, linewidth=1.6, linestyle='-',
                label=f'Predicted Batch_{batch_id}'
            )
        ax.set_title(col, fontsize=13)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration (g/L)')
        ax.grid(True, linestyle='--', color='lightgrey', alpha=0.8)

    axes[7].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc='lower right', bbox_to_anchor=(0.98, 0.02))
    plt.suptitle(f'Predicted vs Observed Trajectories (Test Batches; {admm.transcription_method})', fontsize=14)
    plt.tight_layout()
    output_path = Path(__file__).with_name(f"{admm.transcription_method}_run1_test.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
