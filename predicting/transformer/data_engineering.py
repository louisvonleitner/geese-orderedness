import pandas as pd
import numpy as np

import torch
from predicting.train_tools import mask_geese, normalize_dataframe
from torch.utils.data import TensorDataset, DataLoader


def load_data(features):

    train_data = pd.read_pickle(
        "data/training_data/number_of_geese_data/train_data.pkl"
    )[features]
    val_data = pd.read_pickle("data/training_data/number_of_geese_data/val_data.pkl")[
        features
    ]
    test_data = pd.read_pickle("data/training_data/number_of_geese_data/test_data.pkl")[
        features
    ]

    return train_data, val_data, test_data


def transform_data(data: pd.DataFrame, n, num_created_samples: int):

    # this data is now a numpy array
    data = mask_geese(data, n, num_created_samples)

    # tokenize data
    X_tokenized = np.concatenate(
        [
            [
                data[:, i : i + 3],
                data[:, i + 9 : i + 12],
                data[:, i + 18 : i + 21],
            ]
            for i in range(0, 9, 3)
        ],
        axis=0,
    ).transpose((1, 2, 0))

    # create CLS token
    cls_token = np.zeros((data.shape[0], 1, 9))
    # add CLS token
    X_tokenized = np.concatenate([cls_token, X_tokenized], axis=1)

    print(f"Tokenized data of shape: {X_tokenized.shape}", flush=True)

    y_data = data[:, -1]

    X_data = torch.from_numpy(X_tokenized).float()
    y_data = torch.from_numpy(y_data).float()

    return X_data, y_data


def create_dataloaders(batch_size, n, num_created_samples):
    """
    Load data, transform data, turn into dataloader.

    Each in their own function.
    """
    features = ["centered_positions", "velocities", "accelerations", "n_geese"]
    multidim_features = ["centered_positions", "velocities", "accelerations"]

    train_data, val_data, test_data = load_data(features)

    train_data = normalize_dataframe(train_data, features, multidim_features)
    val_data = normalize_dataframe(val_data, features, multidim_features)

    # apply transforms and norms and turn into pytorch tensor
    X_train, y_train = transform_data(train_data, n, num_created_samples)
    X_val, y_val = transform_data(val_data, n, num_created_samples)
    # X_test, y_test = transform_data(test_data, n, num_created_samples)

    # create Datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    # test_dataset = TensorDataset(X_test, y_test)

    # create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader  # , test_loader
