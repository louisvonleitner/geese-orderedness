import optuna
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, max_error, r2_score

from predicting.boosted_trees.boosted_tree_training import prepare_data, save_model


def tree_objective(trial):
    """
    Objective function for Bayesian Optimization.

    BO is a maximization algorithm, but we want to MINIMIZE the loss.
    Therefore, we return the NEGATIVE of the validation loss.
    """
    # ===================================================
    # model architecture
    max_depth = trial.suggest_int("max_depth", 1, 10)

    # ===================================================
    # learning parameters
    lr = trial.suggest_float("lr", 1e-5, 1.0, log=True)
    subsample = trial.suggest_float("subsample", 0.3, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    # l1_reg = trial.suggest_float("l1_reg", 1e-8, 1.0, log=True)
    l1_reg = 0
    l2_reg = trial.suggest_float("l2_reg", 1e-8, 10.0, log=True)

    # ===================================================
    params = {
        "verbosity": 0,
        "device": "cuda:0",
        "tree_method": "hist",
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "max_depth": max_depth,
        "eta": lr,
        "alpha": l1_reg,
        "lambda": l2_reg,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
    }

    dtrain, dval = prepare_data(1)

    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation-mae"
    )

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, "train"), (dval, "validation")],
        callbacks=[pruning_callback],
        early_stopping_rounds=75,
    )

    # evaluate model
    y_pred = bst.predict(dval)
    y_true = dval.get_label()
    y_pred_train = bst.predict(dtrain)
    y_true_train = dtrain.get_label()
    results = {
        "val_mae": mean_absolute_error(y_pred, y_true),
        "val_max_err": max_error(y_pred, y_true),
        "val_r2": r2_score(y_pred, y_true),
        "train_mae": mean_absolute_error(y_pred_train, y_true_train),
        "train_max_err": max_error(y_pred_train, y_true_train),
        "train_r2": r2_score(y_pred_train, y_true_train),
    } | params
    save_model(bst, results)

    return bst.best_score
