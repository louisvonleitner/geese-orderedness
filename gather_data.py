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


def compute_crosswind_func(
    average_flock_velocity,
    camera_facing_direction_deg,
    wind_coming_from_direction_deg,
    wind_speed_ms,
):
    """
    Compute crosswind acting on a bird flock when:
    - Bird velocity is expressed in camera coordinates
    - Wind direction is expressed in world compass degrees (N/E/S/W)

    Coordinate conventions:
    - Camera y-axis = direction the camera faces (forward)
    - Camera x-axis = right-left direction in the image
    - Wind direction is 'coming from' (meteorological convention)

    returns crosswind in m/s (positive <==> wind from the left, negative <==> wind from the right)
    """

    # -------------------------
    # 1. Extract flock velocity in camera XY
    # -------------------------
    flock_xy = np.array(
        [average_flock_velocity[0], average_flock_velocity[1]], dtype=float
    )

    flock_speed = np.linalg.norm(flock_xy)
    if flock_speed == 0:
        raise Exception("Average velocity of bird flock is 0!")
        return 0.0

    flock_dir_unit = flock_xy / flock_speed

    # -------------------------
    # 2. Build wind vector in WORLD coordinates
    # -------------------------
    # Meteorological convention:
    #   0° = wind from North  → blowing toward South
    #   90° = wind from East → blowing toward West
    theta = np.deg2rad(wind_coming_from_direction_deg)

    wind_world = wind_speed_ms * np.array(
        [np.sin(theta), -np.cos(theta)]  # +x = East  # +y = South
    )

    # -------------------------
    # 3. Rotate wind vector into CAMERA coordinate system
    # -------------------------
    # Camera facing direction (world frame):
    #   0° = North, 90° = East
    # To convert world → camera:
    #   rotate by  -camera_facing_direction
    phi = np.deg2rad(camera_facing_direction_deg)

    R_world_to_camera = np.array(
        [[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]
    )

    wind_camera = R_world_to_camera @ wind_world

    # -------------------------
    # 4. Compute perpendicular direction to flock motion
    # -------------------------
    perpendicular_unit = np.array([-flock_dir_unit[1], flock_dir_unit[0]])

    # -------------------------
    # 5. Crosswind = projection of wind onto perpendicular
    # -------------------------
    crosswind_ms = np.dot(wind_camera, perpendicular_unit)

    return crosswind_ms


def compute_crosswind(foldername: str):

    # Get wind data for trjs
    recording_df = pd.read_excel("data/TABLE-2014-2023.xlsx", usecols="A,K,L,M")

    wind_frame_id = foldername.split("E", 1)[0]

    # get windspeed from file
    wind_speed = recording_df.loc[recording_df["ID"] == wind_frame_id, "WIND[m/s]"]
    if not wind_speed.empty:
        wind_speed = wind_speed.iloc[0]
    else:
        wind_speed = np.nan
        return wind_speed

    if not np.isnan(wind_speed):
        # get directional information from excel sheet
        wind_direction_angle = recording_df.loc[
            recording_df["ID"] == wind_frame_id, "WIND DIR[deg]"
        ].iloc[0]
        camera_direction_angle = recording_df.loc[
            recording_df["ID"] == wind_frame_id, "CAMERA DIR[DEG]"
        ].iloc[0]

        # load average velocity vector
        average_velocities = np.load(
            "data/" + foldername + "/average_velocity_vectors.npy"
        )
        if len(average_velocities) == 0:
            return np.nan

        average_velocity = np.mean(average_velocities, axis=0)

        if type(average_velocity) == np.ndarray:

            crosswind = compute_crosswind_func(
                average_flock_velocity=average_velocity,
                camera_facing_direction_deg=camera_direction_angle,
                wind_coming_from_direction_deg=wind_direction_angle,
                wind_speed_ms=wind_speed,
            )
            return crosswind

        else:
            return np.nan


features = [
    "trj_name",
    "n_frames",
    "crosswind_speed",
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
    "first_pca_component",
    "second_pca_component",
    "first_pca_component_velocity_alignment",
    "second_pca_component_velocity_alignment",
]
metric_dfs = {}
for metric in data_metrics:
    metric_dfs[metric] = pd.DataFrame([], columns=features)

mean_list = []


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

        # compute crosswind
        crosswind = compute_crosswind(foldername)
        means["crosswind_speed"] = crosswind

        # read data points into numpy arrays and calculate mean and other metrics
        for metric in data_metrics:
            if not nan_metric:
                if (
                    metric == "first_pca_component"
                    or metric == "second_pca_component"
                    or metric == "first_pca_component_velocity_alignment"
                    or metric == "second_pca_component_velocity_alignment"
                ):
                    values = read_metric_csv_into_list(
                        "data/" + foldername + "/PCA_velocity_metric_values.csv"
                    )
                    if metric == "velocity_pca_first_component":
                        values = [j[0] for j in values]
                    elif metric == "second_pca_component":
                        values = [j[1] for j in values]
                    elif metric == "first_pca_component_velocity_alignment":
                        values = [j[2] for j in values]
                    elif metric == "second_pca_component_velocity_alignment":
                        values = [j[3] for j in values]
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
                        "values": values.tolist(),
                        "mean": mean,
                        "median": median,
                        "std_dev": std_dev,
                        "maximum": maximum,
                        "minimum": minimum,
                    }
                    metric_dfs[metric].loc[len(metric_dfs[metric])] = row

        if not nan_metric:
            mean_list.append(means)


mean_df = pd.DataFrame(
    mean_list,
    columns=[
        "trj_name",
        "n_geese",
        "n_frames",
        "crosswind_speed",
        "normalized_velocity_alignment",
        "normalized_velocity_alignment_std_dev",
        "velocity_deviation",
        "velocity_deviation_std_dev",
        "sidewise_acceleration_deviation",
        "sidewise_acceleration_deviation_std_dev",
        "longitudinal_acceleration_deviation",
        "longitudinal_acceleration_deviation_std_dev",
        "first_pca_component",
        "first_pca_component_std_dev",
        "second_pca_component",
        "second_pca_component_std_dev",
        "first_pca_component_velocity_alignment",
        "first_pca_component_velocity_alignment_std_dev",
        "second_pca_component_velocity_alignment",
        "second_pca_component_velocity_alignment_std_dev",
    ],
)


mean_df.to_csv("data/metric_means.csv", mode="w")
for metric in data_metrics:
    metric_dfs[metric].to_csv("data/" + metric + "_df.csv")
