import optuna
import numpy as np
from predicting.transformer.training import training


def transformer_objective(trial):
    # ===================================================
    # model architecture
    n_layers = trial.suggest_int("n_layers", 2, 16)
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    head_dim = trial.suggest_categorical("head_dim", [4, 8, 16])

    # Calculate d_model based on n_heads * head_dim
    d_model = n_heads * head_dim
    trial.set_user_attr("d_model", d_model)

    # ===================================================
    # learning parameters
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.0001, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    beta_1 = trial.suggest_float("beta_1", 0.7, 1.0)
    beta_2 = trial.suggest_float("beta_2", 0.9, 1.0)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # min lr ratio
    min_lr = min_lr_ratio * lr
    trial.set_user_attr("min_lr", min_lr)

    # ===================================================
    params = {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "lr": lr,
        "min_lr": min_lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "dropout": dropout,
    }

    best_val_loss = training(params, trial)

    return best_val_loss
