import os
import pandas as pd
from tqdm import trange

from data_engineering.data_engineering import engineer_trajectory_data


def execute_data_engineering(trj_folder: str, saving_folder: str):
    """
    Iterates through trajectory files, engineers features, and saves them.
    """

    # 1. Ensure the saving folder exists; create it if it doesn't
    os.makedirs(saving_folder, exist_ok=True)

    # 2. Iterate over actual file names in the directory
    # os.listdir returns a list of filenames like ['file1.csv', 'file2.csv']
    files = os.listdir(trj_folder)
    for i in range(len(files)):

        print(f"Starting engineering on {i} / {len(files)}...")
        filename = files[i]

        # Optional: Filter to ensure you only process specific file types (e.g., .csv or .parquet)
        if not filename.endswith(".trj"):
            continue

        # 3. Construct the full input path
        input_path = os.path.join(trj_folder, filename)

        print(f"Processing: {filename}...")

        try:
            # 4. Run engineering logic (passing the full path)
            # Ensure engineer_trajectory_data function handles loading the file
            engineered_df = engineer_trajectory_data(input_path)

            # 5. Construct full output path
            output_path = os.path.join(saving_folder, filename)

            # 6. Save the dataframe
            engineered_df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

        # engineered_df = engineer_trajectory_data(input_path)

        # # 5. Construct full output path
        # output_path = os.path.join(saving_folder, filename)

        # # 6. Save the dataframe
        # engineered_df.to_csv(output_path, index=False)

    print("Done!")


# --- Usage ---
trj_folder_path = "data/trajectory_data"
saving_folder_path = "data/prepared_trajectory_data"

execute_data_engineering(trj_folder_path, saving_folder_path)
