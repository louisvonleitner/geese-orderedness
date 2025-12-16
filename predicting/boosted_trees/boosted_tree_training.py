import numpy as np
import pandas as pd
import os

import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, max_error, r2_score

from predicting.train_tools import load_data, mask_geese
import torch
from datetime import datetime


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
    train_data = pd.read_pickle("data/training_data/train_data.pkl")
    validation_data = pd.read_pickle("data/training_data/validation_data.pkl")
    test_data = pd.read_pickle("data/training_data/test_data.pkl")

    # cover random birds of the data except for n
    n = 3
    train_data = mask_geese(train_data, n)  # these are now numpy arrays
    test_data = mask_geese(test_data, n)
    validation_data = mask_geese(validation_data, n)

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    dtrain = xgb.DMatrix(X_train, label=y_train)

    X_val = validation_data[:, :-1]
    y_val = validation_data[:, -1]
    dval = xgb.DMatrix(X_val, label=y_val)

    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    dtest = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dval, dtest


def train_tree(params: dict, num_boost_rounds: int, dtrain, dval):

    # ================================================================
    # train the model

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    y_true = dval.get_label()
    y_pred = bst.predict(dval)

    res = {
        "mae": mean_absolute_error(y_pred, y_true),
        "max_err": max_error(y_pred, y_true),
        "r2": r2_score(y_pred, y_true),
    }

    # ====================================================================
    return res, bst


def grid_search(filepath):

    print("Reading Data", flush=True)
    # load & prepare data
    dtrain, dval, dtest = prepare_data(filepath)

    # set up grid
    num_boost_roundss = [20000]
    lrs = np.array([0.0001, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007])
    max_depths = [20]

    n_models = len(num_boost_roundss) * len(lrs) * len(max_depths)
    i = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")
    if device != "cuda":
        raise Exception("Not using CUDA!")

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
                    "verbosity": 1,
                    "device": "cuda:1",
                    # learning parameters
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                    "eval_metric": "mae",
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
                res, bst = train_tree(params, num_boost_rounds, dtrain, dval)

                result = {
                    "learning_rate": lr,
                    "num_trees": num_boost_rounds,
                    "max_depth": max_depth,
                }
                result = result | res
                results.append(result)

    results = pd.DataFrame(results)

    os.makedirs("predicting/results", exist_ok=True)

    # Create a string with Year-Month-Day_Hour-Minute
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    results.to_csv(f"predicting/results/boosted_trees_maes_{timestamp}.csv")

    return results


def tree_training_wrapper(filepath):
    print("Reading Data", flush=True)
    # load & prepare data
    dtrain, dval, dtest = prepare_data(filepath)

    # define model params
    # ,learning_rate,num_trees,max_depth,mae,max_err,r2
    # 7            7          0.003       5000         20  1.742615  37.892548  0.958045
    num_boost_rounds = 5000
    lr = 0.003
    max_depth = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    if device != "cuda":
        raise Exception("Not using CUDA!")

    print("Starting Model Training", flush=True)

    params = {
        "verbosity": 1,
        "device": "cuda:1",
        "tree_method": "hist",
        # learning parameters
        "objective": "reg:absoluteerror",
        # learning parameters
        "eval_metric": "mae",
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
    result, bst = train_tree(params, num_boost_rounds, dtrain, dval)

    results = {
        "learning_rate": lr,
        "num_trees": num_boost_rounds,
        "max_depth": max_depth,
    } | result
    print(result)

    os.makedirs("predicting/models/boosted_trees", exist_ok=True)

    # save model
    bst.save_model("predicting/models/boosted_trees/n_geese_from_3_geese.json")
    print("Saved model!")

    return results
