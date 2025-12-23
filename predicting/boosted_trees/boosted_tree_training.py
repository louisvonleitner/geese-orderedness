import numpy as np
import pandas as pd
import os

import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, max_error, r2_score

from predicting.train_tools import load_data, mask_geese, normalize_dataframe
import torch
from datetime import datetime


def prepare_data(num_samples):
    features = [
        "centered_positions",
        "velocities",
        "accelerations",
        # "first_pca_component",
        # "second_pca_component",
        "n_geese",
    ]
    input_features = [
        "centered_positions",
        "velocities",
        "accelerations",
        # "first_pca_component",
        # "second_pca_component",
    ]
    multidim_features = ["centered_positions", "velocities", "accelerations"]
    target_feature = "n_geese"

    # load data
    train_data = pd.read_pickle(
        "data/training_data/number_of_geese_data/train_data.pkl"
    )[features]
    val_data = pd.read_pickle("data/training_data/number_of_geese_data/val_data.pkl")[
        features
    ]
    print(
        f"Number of data points: Train = {len(train_data)}, Validation: {len(val_data)}"
    )

    # normalize data
    train_data = normalize_dataframe(
        df=train_data, features=features, multidim_features=multidim_features
    )
    val_data = normalize_dataframe(
        df=val_data, features=features, multidim_features=multidim_features
    )
    # cover random birds of the data except for n
    n = 3
    # number of created samples per data point (5 (min birds) choose 3 (covered birds) = 10, we take half of that, so 5)
    num_created_samples = num_samples
    # mask data
    print("Masking data", flush=True)
    train_data = mask_geese(
        train_data, n, num_created_samples
    )  # these are now numpy arrays
    val_data = mask_geese(val_data, n, num_created_samples)

    print(
        f"Final number of created data points: Train = {train_data.shape[0]}, Validation: {val_data.shape[0]}"
    )

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    dtrain = xgb.DMatrix(X_train, label=y_train)

    X_val = val_data[:, :-1]
    y_val = val_data[:, -1]
    dval = xgb.DMatrix(X_val, label=y_val)

    return dtrain, dval


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

    y_true_train = dtrain.get_label()
    y_pred_train = bst.predict(dtrain)

    res = {
        "val_mae": mean_absolute_error(y_pred, y_true),
        "val_max_err": max_error(y_pred, y_true),
        "val_r2": r2_score(y_pred, y_true),
        "train_mae": mean_absolute_error(y_pred_train, y_true_train),
        "train_max_err": max_error(y_pred_train, y_true_train),
        "train_r2": r2_score(y_pred_train, y_true_train),
    }

    # ====================================================================
    return res, bst


def grid_search():

    print("Reading Data", flush=True)

    # set up grid
    num_boost_roundss = [2500]
    lrs = np.array([0.001, 0.01, 0.1, 0.3, 0.5])
    max_depths = [1, 3]
    beta_1s = [0]
    beta_2s = [0.5, 1, 5, 10]
    nums_samples = [1]

    n_models = (
        len(num_boost_roundss)
        * len(lrs)
        * len(max_depths)
        * len(beta_1s)
        * len(beta_2s)
        * len(nums_samples)
    )
    i = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")
    if device != "cuda":
        raise Exception("Not using CUDA!")

    results = []

    print("Starting Model Training", flush=True)

    for num_samples in nums_samples:

        # load & prepare data
        dtrain, dval = prepare_data(num_samples)
        for num_boost_rounds in num_boost_roundss:
            for lr in lrs:
                for max_depth in max_depths:
                    for beta_1 in beta_1s:
                        for beta_2 in beta_2s:
                            i += 1
                            print(
                                f"Starting building tree {i} / {n_models}", flush=True
                            )
                            print(
                                f"============================================================",
                                flush=True,
                            )
                            params = {
                                "verbosity": 1,
                                "device": "cuda:0",
                                # learning parameters
                                "tree_method": "hist",
                                "objective": "reg:absoluteerror",
                                "eval_metric": "mae",
                                "eta": lr,
                                # tree parameters
                                "max_depth": max_depth,
                                "subsample": 0.8,
                                "colsample_bytree": 0.7,
                                # regularization
                                "lambda": beta_2,  # L2 reg: Smooths the weights
                                "alpha": beta_1,  # L1 reg: Promotes sparsity (drives some leaf weights to 0)
                            }

                            # train model
                            res, bst = train_tree(
                                params, num_boost_rounds, dtrain, dval
                            )

                            result = {
                                "learning_rate": lr,
                                "num_trees": num_boost_rounds,
                                "max_depth": max_depth,
                                "alpha": beta_1,
                                "lambda": beta_2,
                                "num_created_samples": num_samples,
                            }
                            result = result | res
                            results.append(result)

    results = pd.DataFrame(results)

    os.makedirs("predicting/results/boosted_trees", exist_ok=True)

    # Create a string with Year-Month-Day_Hour-Minute
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    results.to_csv(f"predicting/results/boosted_trees/maes_{timestamp}.csv")

    return results


def save_model(bst, results):
    os.makedirs("predicting/models/boosted_trees", exist_ok=True)
    os.makedirs("predicting/results/boosted_trees", exist_ok=True)

    time_now = datetime.now().strftime("%m%d_%H%M%S")

    # save model
    bst.save_model(
        f"predicting/models/boosted_trees/n_geese_from_3_geese_{time_now}.json"
    )
    print("Saved model!")

    results_df = pd.DataFrame([results])
    results_df.to_csv(
        "predicting/results/boosted_trees/optuna_3_geese_boosted_trees_results.csv",
        mode="a",
        header=False,
        index=False,
    )
    print(results)


def tree_training_wrapper(params):
    # load & prepare data
    dtrain, dval = prepare_data()

    # define model params
    lr = params["lr"]
    num_boost_rounds = params["num_boost_rounds"]
    max_depth = params["max_depth"]
    l2_reg = params["l2_reg"]
    l1_reg = params["l1_reg"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    if device != "cuda":
        raise Exception("Not using CUDA!")

    print("Starting Model Training", flush=True)

    tree_params = {
        "verbosity": 1,
        "device": "cuda:0",
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
        "lambda": l2_reg,  # L2 reg: Smooths the weights
        "alpha": l1_reg,  # L1 reg: Promotes sparsity (drives some leaf weights to 0)
    }

    # train model
    result, bst = train_tree(tree_params, num_boost_rounds, dtrain, dval)

    results = params | result

    save_model(bst, results)

    return results
