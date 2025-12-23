from bayes_opt import BayesianOptimization
import numpy as np

from predicting.boosted_trees.boosted_tree_training import tree_training_wrapper


def tree_objective(
    lr,
    num_boost_rounds,
    max_depth,
    l2_reg,
    l1_reg,
):
    """
    Objective function for Bayesian Optimization.

    BO is a maximization algorithm, but we want to MINIMIZE the loss.
    Therefore, we return the NEGATIVE of the validation loss.
    """
    # ===================================================
    # make values from BO compatible with architecture
    try:
        lr = 10**lr
        num_boost_rounds = int(round(num_boost_rounds))
        max_depth = int(round(max_depth))
        l2_reg = 10**l2_reg
        l1_reg = 10**l1_reg

        # =====================================================
        params = {
            "lr": lr,
            "num_boost_rounds": num_boost_rounds,
            "max_depth": max_depth,
            "l2_reg": l2_reg,
            "l1_reg": l1_reg,
        }

        # Get the validation loss from transformer
        val_loss = tree_training_wrapper(params)

        if np.isnan(val_loss):
            return -9999
        return -val_loss
    except Exception as e:
        return -9999


def run_bayesian_optimization(
    objective_function,
    params,
    n_iter=25,
    init_points=5,
    verbose=2,
):
    """
    Executes the Bayesian Optimization process and saves the progress.
    """

    # 1. Initialize the Optimizer
    optimizer = BayesianOptimization(
        f=objective_function, pbounds=params, random_state=42, verbose=2
    )

    print(f"Starting Bayesian Optimization.")

    # 3. Start the optimization
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    return optimizer


# --- Example of running this function (using prior pbounds) ---

# pbounds = { ... } # Define your search space here
# # Define your objective function here: transformer_objective
# optimizer_result = run_bayesian_optimization_with_saving(
#     objective_function=transformer_objective,
#     pbounds=pbounds,
#     n_iter=25,
#     init_points=5,
# )
