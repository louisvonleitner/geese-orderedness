import numpy as np
import pandas as pd
import os

import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from predicting.train_tools import load_data, mask_geese
import torch


def prepare_data(filepath: str):
    features = [
        "positions",
        "velocities",
        "accelerations",
        "n_geese",
    ]
    input_features = ["positions", "velocities", "accelerations"]
    target_feature = "n_geese"

    # load data
    train_data, test_data, validation_data = load_data(filepath, features)

    # cover random birds of the data except for n
    n = 3
    train_data = mask_geese(train_data, n)
    test_data = mask_geese(test_data, n)
    validation_data = mask_geese(validation_data, n)

    X_train = train_data[input_features].to_numpy()
    y_train = train_data[target_feature].to_numpy()
    dtrain = xgb.DMatrix(X_train, label=y_train)

    X_val = validation_data[input_features].to_numpy()
    y_val = validation_data[target_feature].to_numpy()
    dval = xgb.DMatrix(X_val, label=y_val)

    X_test = test_data[input_features].to_numpy()
    y_test = test_data[target_feature].to_numpy()
    dtest = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dval, dtest


def train_tree(filepath: str, params: dict, dtrain, dval, dtest):

    # ================================================================
    # train the model

    if params["device"] == "cuda":
        bst = xgb.train(
            params,
            dtrain,
            tree_method="gpu_hist",
            num_boost_round=params["num_boost_rounds"],
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=early_stop,
            verbose_eval=50,
        )
    else:
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=params["num_boost_rounds"],
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=early_stop,
            verbose_eval=50,
        )

    y_true = dtest.get_label()
    y_pred = bst.predict(dtest)

    mae = mean_absolute_error(y_pred, y_true)

    # ====================================================================
    return mae


def grid_search(filepath):

    print("Reading Data", flush=True)
    # load & prepare data
    dtrain, dval, dtest = prepare_data(filepath)

    # set up grid
    num_boost_roundss = [250, 500, 750, 1000, 5000]
    lrs = [0.001, 0.01, 0.05, 0.1, 0.2]
    max_depths = [1, 2, 3, 4, 5]

    n_models = len(num_boost_roundss) * len(lrs) * len(max_depths)
    i = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    print("Starting Model Training", flush=True)

    for num_boost_rounds in num_boost_roundss:
        for lr in lrs:
            for max_depth in max_depths:
                i += 1
                print(f"Starting building tree {i} / {n_models}", flush=True)
                print(
                    f"============================================================",
                    flush=True,
                )
                params = {
                    "verbosity": 2,
                    "device": device,
                    # learning parameters
                    "objective": "reg:squarederror",
                    "eval_metric": "mae",
                    "num_boost_rounds": num_boost_rounds,
                    "eta": lr,
                    # tree parameters
                    "max_depth": max_depth,
                    "subsample": 0.8,
                    "colsample_bytree": 0.7,
                    # regularization
                    "lambda": 1,  # L2 reg: Smooths the weights
                    "alpha": 0,  # L1 reg: Promotes sparsity (drives some leaf weights to 0)
                }

                # train model
                mae = train_tree(data, params, dtrain, dval, dtest)

                result = {
                    "mean_absolute_error": mae,
                    "learning_rate": lr,
                    "num_trees": num_boost_rounds,
                    "max_depth": max_depth,
                }
                results.append(result)

    results = pd.DataFrame(results)

    os.mkdirs("predicting/results", exist_ok=True)

    results.to_csv("predicting/results/boosted_trees_maes.csv")

    return results
