import numpy as np
import pandas as pd

import torch

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import ast
import re


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


def n_geese_prediction_mask(initial_row: pd.Series, n_visible_geese):

    row = initial_row.copy()
    n_geese = row["n_geese"]

    random_bird_idx = np.random.choice(
        range(n_geese), n_visible_geese, replace=False, p=None
    )
    # print(random_bird_idx)

    # visible positions
    row["positions"] = np.array(row["positions"])[random_bird_idx]
    # visible velocities
    row["velocities"] = np.array(row["velocities"])[random_bird_idx]
    # visible accelerations
    row["accelerations"] = np.array(row["accelerations"])[random_bird_idx]

    # flatten row
    flat_row = flatten_row(row)

    return flat_row


def mask_geese(df: pd.DataFrame, n: int):

    masked_data = df.apply(
        lambda row: n_geese_prediction_mask(row, n),
        axis=1,
    )

    final_data = np.array(masked_data.tolist())
    return final_data
