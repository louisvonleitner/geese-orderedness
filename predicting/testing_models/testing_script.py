import numpy as np
import pandas as pd

from predicting.train_tools import load_data, n_geese_prediction_mask
from predicting.testing_models.predict import predict_number_of_geese_tree

import matplotlib.pyplot as plt
import xgboost as xgb


def visualize(X: np.ndarray):

    # X has 3 coordinates for every 3 birds for all 3 features
    # need to extract the first 9, which are position
    # the first 3 belong to one bird, and so on
    # make a 2D representation

    # turn one row into numpy array like given to model for prediction
    bird_x = [X[i] for i in range(0, 9, 3)]
    bird_y = [X[i + 1] for i in range(0, 9, 3)]
    bird_z = [X[i + 2] for i in range(0, 9, 3)]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        bird_x,
        bird_y,
        bird_z,
        c="red",
        s=80,
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

    ax.set_title("What the model sees")

    plt.show()


def run_test():
    # load data
    test_data = pd.read_pickle("./data/training_data/test_data.pkl")

    sample = test_data.iloc[
        np.random.choice(range(len(test_data)), 1, replace=False, p=None)
    ].iloc[0]
    print(sample)
    sample = n_geese_prediction_mask(sample, 3)

    y_true = sample[-1]
    X = sample[:-1]
    X_reshaped = X.reshape(1, -1)
    dtest = xgb.DMatrix(X_reshaped, label=[y_true])

    y_pred = predict_number_of_geese_tree(dtest)

    visualize(X)

    print(f"Real value: {y_true} | Model prediction: {y_pred}")
    print(f"Error: {np.abs(y_true - y_pred)}")
