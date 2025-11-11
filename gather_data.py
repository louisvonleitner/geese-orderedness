import numpy as np
import pandas as pd
from tqdm import trange
import os
import gc
import matplotlib.pyplot as plt
import csv

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
                try:
                    values.append(float(item))
                except ValueError:
                    raise ValueError(f"Invalid float value '{item}' in file {filepath}")
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


features = [
    "trj_name",
    "n_frames",
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
    ):
        nan_metric = False

        i += 1
        print(f"Starting analysis {i}/{amount_of_analysises}")

        means = {"trj_name": foldername}

        # determine number of geese in trajectory
        n_geese = get_number_of_geese(foldername)
        means["n_geese"] = n_geese

        # read data points into numpy arrays and calculate mean and other metrics
        for metric in data_metrics:
            if not nan_metric:
                values = read_metric_csv_into_list(
                    "data/" + foldername + "/" + metric + "_values.csv"
                )
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
