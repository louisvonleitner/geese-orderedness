from predicting.transformer.training import training

# from predicting.transformer.bayesian_optimization import (
#     run_bayesian_optimization,
#     transformer_objective,
# )
from predicting.transformer.data_engineering import create_dataloaders
import torch
import time

import optuna
from optuna.pruners import HyperbandPruner, PatientPruner

# from optuna.storages import JournalStorage, JournalFileStorage
from optuna.storages.journal import JournalStorage, JournalFileBackend
from optuna.visualization import plot_optimization_history, plot_param_importances

from predicting.transformer.optuna_optimization import transformer_objective
from predicting.transformer.training import training


# params = {
#     "d_model": 32,
#     "n_heads": 4,
#     "n_layers": 8,
#     "lr": 0.001617,
#     "min_lr": 0.000003,
#     "batch_size": 256,
#     "beta_1": 0.944638,
#     "beta_2": 0.970686,
#     "weight_decay": 0.001657,
# }

# objective_function = transformer_objective

# # train_loader, val_loader, test_loader = create_dataloaders(batch_size=batch_size)
# run_bayesian_optimization(objective_function, pbounds, n_iter=50, init_points=10)


def run_optuna_optimization():
    storage = JournalStorage(
        JournalFileBackend("5_geese_transformer_study_history.log")
    )

    base_hyperband = HyperbandPruner(
        min_resource=20,
        max_resource=100,
        reduction_factor=2,
    )

    combined_pruner = PatientPruner(base_hyperband, patience=10)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name="5_geese_transformer",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
        pruner=combined_pruner,
        direction="minimize",
    )
    study.optimize(transformer_objective, n_trials=50)

    history_fig = plot_optimization_history(study)
    history_fig.write_image("5_geese_transformer_optimization_history.png")
    param_fig = plot_param_importances(study)
    param_fig.write_image("5_geese_transformer_parameter_importance.png")


start_time = time.time()

# training(params)
run_optuna_optimization()
print(f"Script execution took {(time.time() - start_time) / (60 * 60)} hours.")
