import numpy as np
import pandas as pd

from predicting.train_tools import (load_data, n_geese_prediction_mask)
from predicting.testing_models.predict import predict_number_of_geese_tree

import matplotlib.pyplot as plt


def visualize(X: np.ndarray):

    # X has 3 coordinates for every 3 birds for all 3 features
    # need to extract the first 9, which are position
    # the first 3 belong to one bird, and so on
    # make a 2D representation

    # turn one row into numpy array like given to model for prediction
    bird_positions = np.ndarray(np.ndarray([X[i], X[i + 3]] for i in range(3)))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.scatterplot(
        bird_positions,
        ax=ax,
        )
    
    fig.title("What the model sees")

    plt.show()



def run_test():
    # load data
    data_path = "../data/training_data/number_of_geese.pkl"
    features = [
            "positions",
            "velocities",
            "accelerations",
            "n_geese",
        ]
    train_data, validation_data, test_data = load_data(data_path, features)

    sample = test_data.iloc[np.random.choice(
        range(len(test_data)), 1, replace=False, p=None
    )
    ]
    sample = n_geese_prediction_mask(sample, 3)

    y_true = sample[-1]
    X = sample[:-1]

    y_pred = predict_number_of_geese_tree(X)

    visualize(X)

    print(f"Real value: {y_true} | Model prediction: {y_pred}")
    print(f"Error: {np.abs(y_true - y_pred)}")



    



