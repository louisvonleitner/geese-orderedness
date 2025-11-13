import numpy as np
import pandas as pd
from tqdm import trange
import os
import gc
import matplotlib.pyplot as plt
import csv
import re

from data_engineering.metric_calculation import read_trajectory_data, clean_data

# ===================================================================================================
folder_path = "data"
# ===================================================================================================


def read_metric_csv_into_list(filepath):
    values = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if not item:
                    continue

                # Remove np.float64(...) wrappers if present
                item = re.sub(r"np\.float64\(([^)]+)\)", r"\1", item)

                # Try simple float conversion
                try:
                    values.append(float(item))
                    continue
                except ValueError:
                    pass

                # Try tuple parsing, handling np.float64 inside tuples
                if item.startswith("(") and item.endswith(")"):
                    inner = item.strip("()")
                    inner = re.sub(r"np\.float64\(([^)]+)\)", r"\1", inner)
                    try:
                        t = tuple(float(x) for x in inner.split(","))
                        values.append(t)
                        continue
                    except ValueError:
                        pass

                # If all else fails
                raise ValueError(f"Invalid item '{item}' in file {filepath}")

    return values


def get_number_of_geese(foldername):
    # ====================================================================================================
    # setting values to be used from dataset
    file_column_numbers = [0, 1, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18]
    file_column_names = [
        "trj_id",
        "frame",
        "xpos",
        "ypos",
        "zpos",
        "xvel",
        "yvel",
        "zvel",
        "n",
        "xi",
        "eta",
        "zeta",
    ]
    # =====================================================================================================

    # creating dataframes from function
    df, individual_geese_trjs, n_trjs = read_trajectory_data(
        foldername, file_column_numbers, file_column_names
    )

    df, individual_geese_trjs, n_trjs = clean_data(df, individual_geese_trjs)
    return n_trjs


# ===========================================================================================
# Get wind data for trjs

wind_df = pd.read_excel("data/TABLE-2014-2023.xlsx", usecols="A,L,M")

# ================================================================================================


features = [
    "trj_name",
    "n_frames",
    "wind_speed",
    "values",
    "mean",
    "median",
    "std_dev",
    "maximum",
    "minimum",
]

directory_list = os.listdir(folder_path)

amount_of_analysises = len(directory_list) - 1


data_metrics = [
    "normalized_velocity_alignment",
    "velocity_deviation",
    "sidewise_acceleration_deviation",
    "longitudinal_acceleration_deviation",
    "velocity_pca_first_component",
    "velocity_pca_second_component",
]
metric_dfs = {}
for metric in data_metrics:
    metric_dfs[metric] = pd.DataFrame([], columns=features)

mean_df = pd.DataFrame(
    [],
    columns=[
        "trj_name",
        "n_geese",
        "n_frames",
        "wind_speed",
        "normalized_velocity_alignment",
        "normalized_velocity_alignment_std_dev",
        "velocity_deviation",
        "velocity_deviation_std_dev",
        "sidewise_acceleration_deviation",
        "sidewise_acceleration_deviation_std_dev",
        "longitudinal_acceleration_deviation",
        "longitudinal_acceleration_deviation_std_dev",
        "velocity_pca_first_component",
        "velocity_pca_first_component_std_dev",
        "velocity_pca_second_component",
        "velocity_pca_second_component_std_dev",
    ],
)

i = 0
for foldername in directory_list:
    if (
        foldername != "trajectory_data"
        and foldername != "NOTES.txt"
        and ".csv" not in foldername
        and ".xlsx" not in foldername
    ):
        nan_metric = False

        i += 1
        print(f"Starting analysis {i}/{amount_of_analysises}")

        means = {"trj_name": foldername}

        # determine number of geese in trajectory
        n_geese = get_number_of_geese(foldername)
        means["n_geese"] = n_geese

        wind_frame_id = foldername.split("E", 1)[0]
        wind_speed = wind_df.loc[wind_df["ID"] == wind_frame_id, "WIND[m/s]"]
        if not wind_speed.empty:
            wind_speed = wind_speed.iloc[0]
        else:
            wind_speed = np.nan

        means["wind_speed"] = wind_speed
        # read data points into numpy arrays and calculate mean and other metrics
        for metric in data_metrics:
            if not nan_metric:
                if (
                    metric == "velocity_pca_first_component"
                    or metric == "velocity_pca_second_component"
                ):
                    values = read_metric_csv_into_list(
                        "data/" + foldername + "/PCA_velocity_metric_values.csv"
                    )
                    if metric == "velocity_pca_first_component":
                        values = [j[0] for j in values]
                    else:
                        values = [j[1] for j in values]
                # if not PCA metric
                else:
                    values = read_metric_csv_into_list(
                        "data/" + foldername + "/" + metric + "_values.csv"
                    )

                # turn into numpy array and remove nans
                values = np.array(values)
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    nan_metric = True
                    break

                else:
                    trj_name = foldername
                    values = np.array(values)
                    mean = np.mean(values)
                    median = np.median(values)
                    std_dev = np.std(values)
                    maximum = np.max(values)
                    minimum = np.min(values)

                    means[metric] = mean
                    means[f"{metric}_std_dev"] = std_dev
                    means["n_frames"] = len(values)

                    row = {
                        "trj_name": trj_name,
                        "n_frames": len(values),
                        "values": values,
                        "mean": mean,
                        "median": median,
                        "std_dev": std_dev,
                        "maximum": maximum,
                        "minimum": minimum,
                    }
                    metric_dfs[metric].loc[len(metric_dfs[metric])] = row

        if not nan_metric:
            mean_df.loc[len(mean_df)] = means


mean_df.to_csv("data/metric_means.csv", mode="w")
for metric in data_metrics:
    metric_dfs[metric].to_csv("data/" + metric + "_df.csv")
