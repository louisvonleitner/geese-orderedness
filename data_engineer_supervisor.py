import os
import pandas as pd
from tqdm import trange
import traceback

from data_engineering.data_engineering import engineer_trajectory_data


def execute_data_engineering(trj_folder: str, saving_path: str):
    """
    Iterates through trajectory files, engineers features, and saves them.
    """

    created_dfs = []

    # iterate over file names in the target directory
    files = os.listdir(trj_folder)
    for i in range(len(files)):

        print(f"Starting engineering on {i} / {len(files)}...")
        filename = files[i]

        # ensure to only process .trj file types
        if not filename.endswith(".trj"):
            continue

        # construct full input path
        input_path = os.path.join(trj_folder, filename)

        print(f"Processing: {filename}...")

        # Run engineering logic
        engineered_df = engineer_trajectory_data(input_path)

        if type(engineered_df) is pd.DataFrame:
            created_dfs.append(engineered_df)

    full_df = pd.concat(created_dfs)
    full_df.to_pickle(saving_path)
    print("Done!")


# --- Usage ---
train_trj_folder_path = "data/training_data/number_of_geese_data/train_data"
val_trj_folder_path = "data/training_data/number_of_geese_data/val_data"
test_trj_folder_path = "data/training_data/number_of_geese_data/test_data"
train_saving_path = "data/training_data/number_of_geese_data/train_data.pkl"
val_saving_path = "data/training_data/number_of_geese_data/val_data.pkl"
test_saving_path = "data/training_data/number_of_geese_data/test_data.pkl"

execute_data_engineering(train_trj_folder_path, train_saving_path)
execute_data_engineering(val_trj_folder_path, val_saving_path)
execute_data_engineering(test_trj_folder_path, test_saving_path)
