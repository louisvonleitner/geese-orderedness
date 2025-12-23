from predicting.boosted_trees.bayesian_optimization import (
    run_bayesian_optimization,
    tree_objective,
)
import time

import optuna
from optuna.pruners import HyperbandPruner, PatientPruner

# from optuna.storages import JournalStorage, JournalFileStorage
from optuna.storages.journal import JournalStorage, JournalFileBackend
from optuna.visualization import plot_optimization_history, plot_param_importances

from predicting.boosted_trees.optuna_optimization import tree_objective
from predicting.boosted_trees.boosted_tree_training import grid_search


def run_optuna_optimization():

    storage = JournalStorage(JournalFileBackend("3_geese_tree_study_history.log"))

    base_hyperband = HyperbandPruner(
        min_resource=100,
        max_resource=10000,
        reduction_factor=2,
    )

    combined_pruner = PatientPruner(base_hyperband, patience=100)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name="3_geese_tree_study",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
        pruner=combined_pruner,
        direction="minimize",
    )
    study.optimize(tree_objective, n_trials=50)

    history_fig = plot_optimization_history(study)
    history_fig.write_image("3_geese_tree_optimization_history.png")
    param_fig = plot_param_importances(study)
    param_fig.write_image("3_geese_tree_parameter_importance.png")


start_time = time.time()
run_optuna_optimization()
# grid_search()

print(f"Script execution took {(time.time() - start_time) / (60 * 60)} hours.")
