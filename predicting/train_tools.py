import numpy as np
import pandas as pd
import ast
import re
import pickle

import torch

from sklearn.model_selection import train_test_split


def load_data(filepath, features: list, temporal=False):
    """
    Load training dataset properly and split into train, test and validation set.

    Load training dataset into a pandas DataFrame.
    Split into train, test and validation depending on data format.

    Args:
        filepath (str): The path to the CSV data file.
        temporal (bool): If True, performs a time-aware split (not implemented here
                         but logic for a random split is provided).

    Returns:
        tuple: A tuple containing the (train, test, validation) pandas DataFrames.
    """

    df = pd.read_pickle(filepath)

    df = df[features]

    # print(type(df["positions"].iloc[0]), flush=True)

    RANDOM_STATE = 42

    # --- Split Logic ---

    # 1. Split off the Test set first (e.g., 20% total, or 10% here)
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.20,  # 10% for test
        random_state=RANDOM_STATE,
        shuffle=not temporal,  # Don't shuffle if temporal data
    )

    # We want 10% for validation, so we take 10% of the remaining 90% (10 / 90 = 0.111...)
    validation_size_ratio = 0.10 / (
        1.0 - 0.10
    )  # 0.111... to get 10% of original data size

    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=validation_size_ratio,
        random_state=RANDOM_STATE,
        shuffle=not temporal,  # Don't shuffle if temporal data
    )

    # Note: If temporal=True, you would typically use a custom time-based split
    # instead of the random split above.

    print(
        f"Data Split: Train={len(train_df)} ({len(train_df)/len(df):.1%}), Validation={len(validation_df)} ({len(validation_df)/len(df):.1%}), Test={len(test_df)} ({len(test_df)/len(df):.1%})",
        flush=True,
    )

    return train_df, validation_df, test_df


def normalize_dataframe(df: pd.DataFrame, features, multidim_features):

    with open("data/training_data/number_of_geese_data/data_norm.pkl", "rb") as f:
        norm_dict = pickle.load(f)

    means = norm_dict["means"]
    stds = norm_dict["stds"]

    for feature in features:
        m = means[feature]
        s = stds[feature]

        if feature in multidim_features:
            # Use apply to normalize each row's array/list
            df[feature] = df[feature].apply(lambda x: (np.array(x) - m) / (s + 1e-8))
        else:
            # Handle standard scalar columns
            df[feature] = (df[feature] - m) / (s + 1e-8)

    return df


def get_std_n_geese():

    with open("data/training_data/number_of_geese_data/data_norm.pkl", "rb") as f:
        norm_dict = pickle.load(f)

    means = norm_dict["means"]
    stds = norm_dict["stds"]

    return stds["n_geese"]


def inverse_normalization_n_geese(row: pd.Series):
    """
    row has form: [centered_positions], [velocities], [accelerations], [n_geese]
    """
    with open("data/training_data/number_of_geese_data/data_norm.pkl", "rb") as f:
        norm_dict = pickle.load(f)

    means = norm_dict["means"]
    stds = norm_dict["stds"]

    features = row.index
    multidim_features = ["centered_positions", "velocities", "accelerations"]

    for feature in features:
        m = means[feature]
        s = stds[feature]

        row[feature] = np.array(row[feature]) * (s + 1e-8) + m

    return row


def flatten_row(row: pd.Series) -> np.ndarray:
    """
    Flattens a complex Pandas Series (row) into a single 1D NumPy array.
    It flattens any nested NumPy arrays found within the row, while
    preserving simple scalar values.
    """
    flat_features = []

    for entry in row.values:

        #  simple scalar
        if isinstance(entry, (float, int, np.int64)):
            flat_features.append(entry)

        # nested numpy array
        elif isinstance(entry, np.ndarray):
            # Flatten the nested array and extend the features list
            # We use .tolist() to convert the flattened array for easy list extension
            flat_features.extend(entry.flatten().tolist())

        # list
        elif isinstance(entry, list):
            # If a list is found, assume it needs to be flattened too
            flat_features.extend(entry)

    return np.array(flat_features)


def n_geese_prediction_mask(
    initial_row: pd.Series, n_visible_geese, num_created_samples=5
):
    rows = []
    # 5 (min n birds) choose 3 = 10, we take 5 samples per row
    for i in range(num_created_samples):
        row = initial_row.copy()

        if len(row["centered_positions"]) != len(row["velocities"]) or len(
            row["velocities"]
        ) != len(row["accelerations"]):
            raise Exception(
                "Missmatch in number of positions, velocities and accelerations!"
            )

        # Choose random geese
        random_bird_idx = np.random.choice(
            range(len(row["centered_positions"])),
            n_visible_geese,
            replace=False,
            p=None,
        )
        # print(random_bird_idx)

        # visible positions
        row["centered_positions"] = np.array(row["centered_positions"])[random_bird_idx]
        # visible velocities
        row["velocities"] = np.array(row["velocities"])[random_bird_idx]
        # visible accelerations
        row["accelerations"] = np.array(row["accelerations"])[random_bird_idx]

        # flatten row
        flat_row = flatten_row(row)

        rows.append(flat_row)

    return rows


def mask_geese(df: pd.DataFrame, n: int, num_created_samples: int):

    masked_data = df.apply(
        lambda row: n_geese_prediction_mask(row, n, num_created_samples),
        axis=1,
    )

    exploded_data = masked_data.explode()

    final_data = np.vstack(exploded_data.tolist())

    return final_data
