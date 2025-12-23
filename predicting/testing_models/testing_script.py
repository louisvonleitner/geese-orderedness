import numpy as np
import pandas as pd

from predicting.train_tools_legacy import load_data, n_geese_prediction_mask, mask_geese
from predicting.testing_models.predict import (
    predict_number_of_geese_small_tree,
    predict_number_of_geese_big_tree,
)

import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, max_error, r2_score
import time


def visualize(X: np.ndarray):

    # X has 3 coordinates for every 3 birds for all 3 features
    # need to extract the first 9, which are position
    # the first 3 belong to one bird, and so on
    # make a 2D representation

    # turn one row into numpy array like given to model for prediction
    bird_x = [X[i] for i in range(0, 9, 3)]
    bird_y = [X[i + 1] for i in range(0, 9, 3)]
    bird_z = [X[i + 2] for i in range(0, 9, 3)]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    ax.scatter(
        bird_x,
        bird_y,
        bird_z,
        c="red",
        s=40,
    )

    ax2.scatter(
        bird_x,
        bird_y,
        c="red",
        s=40,
    )

    # concatenate lists
    all_coords = bird_x + bird_y + bird_z

    # Calculate the maximum range in any dimension
    max_range = np.array([max(all_coords) - min(all_coords)])

    # Calculate the center point for each axis
    mid_x = (max(bird_x) + min(bird_x)) / 2
    mid_y = (max(bird_y) + min(bird_y)) / 2
    mid_z = (max(bird_z) + min(bird_z)) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.set_title("What the model sees 3D")

    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)

    ax2.grid("lightgrey")

    ax2.set_xlabel("")
    ax2.set_ylabel("")

    ax2.set_title("What the model sees top view 2D")

    plt.tight_layout()
    plt.savefig("model_view.png", dpi=300)
    plt.close()


def run_test():
    # load data
    test_data = pd.read_pickle("./data/training_data/test_data.pkl")

    sample = test_data.iloc[
        np.random.choice(range(len(test_data)), 1, replace=False, p=None)
    ].iloc[0]
    sample = n_geese_prediction_mask(sample, 3)

    y_true = sample[-1]
    y_sample = sample[-1]
    X_1 = sample[:-1]
    positions = sample[0:9]
    velocities = sample[9:18]
    accelerations = sample[18:27]

    order_2 = [2, 3, 1]
    order_3 = [3, 1, 2]

    X_2 = np.concatenate(
        (
            positions[3:6],
            positions[6:9],
            positions[0:3],
            velocities[3:6],
            velocities[6:9],
            velocities[0:3],
            accelerations[3:6],
            accelerations[6:9],
            accelerations[0:3],
        )
    )
    X_3 = np.concatenate(
        (
            positions[6:9],
            positions[0:3],
            positions[3:6],
            velocities[6:9],
            velocities[0:3],
            velocities[3:6],
            accelerations[6:9],
            accelerations[0:3],
            accelerations[3:6],
        )
    )

    X_1_reshaped = X_1.reshape(1, -1)
    X_2_reshaped = X_2.reshape(1, -1)
    X_3_reshaped = X_3.reshape(1, -1)

    dtest_1 = xgb.DMatrix(X_1_reshaped, label=[y_true])
    dtest_2 = xgb.DMatrix(X_2_reshaped, label=[y_true])
    dtest_3 = xgb.DMatrix(X_3_reshaped, label=[y_true])

    print("small model predicting")
    start = time.time()
    y_pred_1_small, small_total_parameters = predict_number_of_geese_small_tree(dtest_1)
    small_model_time = time.time() - start
    y_pred_2_small, small_total_parameters = predict_number_of_geese_small_tree(dtest_2)
    y_pred_3_small, small_total_parameters = predict_number_of_geese_small_tree(dtest_3)

    print("big model predicting")
    start = time.time()
    y_pred_1_big, big_total_parameters = predict_number_of_geese_big_tree(dtest_1)
    big_model_time = time.time() - start
    y_pred_2_big, big_total_parameters = predict_number_of_geese_big_tree(dtest_2)
    y_pred_3_big, big_total_parameters = predict_number_of_geese_big_tree(dtest_3)

    visualize(X_1)

    print("============================================================")
    print("Small model:")
    print(f"Compute time for one value: {small_model_time} seconds")
    print(f"total parameter count: {small_total_parameters}")
    print(
        f"Real value: {y_true} | Model predictions: {y_pred_1_small}, {y_pred_2_small}, {y_pred_3_small}"
    )
    print(
        f"Errors: {np.abs(y_true - y_pred_1_small)}, {np.abs(y_true - y_pred_2_small)}, {np.abs(y_true - y_pred_3_small)}"
    )
    # test_masked = mask_geese(test_data, 3)
    # X = test_masked[:, :-1]
    # y_true = test_masked[:, -1]

    # dtest = xgb.DMatrix(X, label=y_true)

    # y_pred, n_params = predict_number_of_geese_small_tree(dtest)

    # mae = mean_absolute_error(y_true, y_pred)
    # max_e = max_error(y_true, y_pred)
    # r2 = r2_score(y_true, y_pred)

    # print(f"On whole test dataset:")
    # print(f"Mean absolute error: {mae}")
    # print(f"Max absolute error: {max_e}")
    # print(f"Explained Variance (r2 score): {r2}")

    print("============================================================")
    print("Big model:")
    print(f"total parameter count: {big_total_parameters}")
    print(f"Compute time for one value: {big_model_time} seconds")
    print(
        f"Real value: {y_sample} | Model predictions: {y_pred_1_big}, {y_pred_2_big}, {y_pred_3_big}"
    )
    print(
        f"Errors: {np.abs(y_sample - y_pred_1_big)}, {np.abs(y_sample - y_pred_2_big)}, {np.abs(y_sample - y_pred_3_big)}"
    )

    # y_pred, n_params = predict_number_of_geese_big_tree(dtest)

    # mae = mean_absolute_error(y_true, y_pred)
    # max_e = max_error(y_true, y_pred)
    # r2 = r2_score(y_true, y_pred)

    # print(f"On whole test dataset:")
    # print(f"Mean absolute error: {mae}")
    # print(f"Max absolute error: {max_e}")
    # print(f"Explained Variance (r2 score): {r2}")
