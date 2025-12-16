import pandas as pd
import numpy as np

import torch
from predicting.train_tools import mask_geese
from torch.utils.data import TensorDataset, DataLoader


def load_data():

    train_data = pd.read_pickle("data/training_data/train_data_centered_pos.pkl")
    validation_data = pd.read_pickle(
        "data/training_data/validation_data_centered_pos.pkl"
    )
    test_data = pd.read_pickle("data/training_data/test_data_centered_pos.pkl")

    return train_data, validation_data, test_data


def generate_norm(train_data):

    columns = train_data.columns
    means = {}
    stds = {}
    dimensions = {"x": 0, "y": 1, "z": 2}

    for column in columns:
        if column in ["centered_positions", "velocities", "accelerations"]:

            # extract column
            column_data = train_data[column].copy().to_numpy()
            flat_column = []
            for array in column_data:
                for v in array:
                    flat_column.append(v)

            # this is now an array full of 3 dimensional vectors
            flat_column = np.array(flat_column)
            mean_per_position = np.mean(flat_column, axis=0)
            std_per_position = np.std(flat_column, axis=0)

            means[column] = mean_per_position
            stds[column] = std_per_position

        else:
            mean = train_data[column].mean()
            std = train_data[column].std()

            means[column] = mean
            stds[column] = std

    norm = {"means": means, "stds": stds}
    norm_df = pd.DataFrame(norm)
    norm_df.to_csv("data/training_data/norm.csv")

    return norm


def transform_data(norm: dict, data: pd.DataFrame):

    means = norm["means"]
    stds = norm["stds"]
    means = np.array(
        [
            means["centered_positions"][0],
            means["centered_positions"][1],
            means["centered_positions"][2],
            means["centered_positions"][0],
            means["centered_positions"][1],
            means["centered_positions"][2],
            means["centered_positions"][0],
            means["centered_positions"][1],
            means["centered_positions"][2],
            means["velocities"][0],
            means["velocities"][1],
            means["velocities"][2],
            means["velocities"][0],
            means["velocities"][1],
            means["velocities"][2],
            means["velocities"][0],
            means["velocities"][1],
            means["velocities"][2],
            means["accelerations"][0],
            means["accelerations"][1],
            means["accelerations"][2],
            means["accelerations"][0],
            means["accelerations"][1],
            means["accelerations"][2],
            means["accelerations"][0],
            means["accelerations"][1],
            means["accelerations"][2],
            means["n_geese"],
        ]
    )
    stds = np.array(
        [
            stds["centered_positions"][0],
            stds["centered_positions"][1],
            stds["centered_positions"][2],
            stds["centered_positions"][0],
            stds["centered_positions"][1],
            stds["centered_positions"][2],
            stds["centered_positions"][0],
            stds["centered_positions"][1],
            stds["centered_positions"][2],
            stds["velocities"][0],
            stds["velocities"][1],
            stds["velocities"][2],
            stds["velocities"][0],
            stds["velocities"][1],
            stds["velocities"][2],
            stds["velocities"][0],
            stds["velocities"][1],
            stds["velocities"][2],
            stds["accelerations"][0],
            stds["accelerations"][1],
            stds["accelerations"][2],
            stds["accelerations"][0],
            stds["accelerations"][1],
            stds["accelerations"][2],
            stds["accelerations"][0],
            stds["accelerations"][1],
            stds["accelerations"][2],
            stds["n_geese"],
        ]
    )

    # this data is now a numpy array
    data = mask_geese(data, 3)

    # norm data
    normed_data = data - means / stds

    # tokenize data
    X_tokenized = np.concatenate(
        [
            [
                normed_data[:, i : i + 3],
                normed_data[:, i + 9 : i + 12],
                normed_data[:, i + 18 : i + 21],
            ]
            for i in range(0, 9, 3)
        ],
        axis=0,
    ).transpose((1, 2, 0))

    # create CLS token
    cls_token = np.zeros((normed_data.shape[0], 1, 9))
    # add CLS token
    X_tokenized = np.concatenate([cls_token, X_tokenized], axis=1)

    print(X_tokenized.shape)

    y_data = normed_data[:, -1]

    X_data = torch.from_numpy(X_tokenized).float()
    y_data = torch.from_numpy(y_data).float()

    return X_data, y_data


def create_dataloaders(batch_size):
    """
    Load data, transform data, turn into dataloader.

    Each in their own function.
    """

    train_data, validation_data, test_data = load_data()

    # calculate norm from train data
    norm = generate_norm(train_data)

    # apply transforms and norms and turn into pytorch tensor
    X_train, y_train = transform_data(norm, train_data)
    X_val, y_val = transform_data(norm, validation_data)
    X_test, y_test = transform_data(norm, test_data)

    # create Datasets
    train_dataset = TensorDataset(X_train, y_train)
    validation_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
