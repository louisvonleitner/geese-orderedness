from bayes_opt import BayesianOptimization
import numpy as np
from predicting.transformer.training import training


# --- 2. The Objective Function for Bayesian Optimization ---
# def setup_bo(train_loader, val_loader):
def transformer_objective(
    lr,
    min_lr_ratio,
    batch_size,
    n_epochs,
    warmup_ratio,
    n_heads,
    n_layers,
    d_model,
    beta_1,
    beta_2,
    weight_decay,
    epsilon,
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
        min_lr = min_lr_ratio * lr
        batch_size = int(round(batch_size)) * 64

        n_epochs = int(round(n_epochs))

        if n_heads < 2:
            n_heads = 2
        else:
            n_heads = 2 * int(round(n_heads / 2))

        n_layers = int(round(n_layers))

        d_model = n_heads * int(round(d_model / n_heads))
        if d_model == 0:
            d_model = n_heads

        # =====================================================
        params = {
            "lr": lr,
            "min_lr": min_lr,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "warmup_ratio": warmup_ratio,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_model": d_model,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "epsilon": epsilon,
            "weight_decay": weight_decay,
        }

        # Get the validation loss from transformer
        val_loss = training(params)

        if np.isnan(val_loss):
            return -9999
        return -val_loss
    except Exception as e:
        print(e)
        return -9999


def run_bayesian_optimization(
    objective_function,
    params,
    n_iter=50,
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
