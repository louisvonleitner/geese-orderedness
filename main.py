import numpy as np
import pandas as pd
from tqdm import trange
import os
import gc
import matplotlib.pyplot as plt

# project's own imports
from data_engineering.run_analysis import trajectory_analysis
from data_engineering.metric_calculation import (
    # boltzmann_metric,
    # inverse_exponential_distance_metric,
    # calculate_velocity_deviation,
    # calculate_velocity_alignment,
    # calculate_longitudinal_acceleration_deviation,
    # calculate_sidewise_acceleration_deviation,
    calculate_velocity_PCA,
    gaussian_entropy,
    flight_deviations,
)


# ===================================================================================================
folder_path = "data/trajectory_data"
# ===================================================================================================

amount_of_analysises = len(os.listdir(folder_path)) - 1

i = 0
for filename in os.listdir(folder_path):
    i += 1

    order_metrics = []

    # PCA velocity metrics
    PCA_velocity_metric = {
        "name": "PCA_velocity_metric",
        "function": calculate_velocity_PCA,
        "values": [],
        "color": "blue",
        "value_space": (
            [0, 10],
            [0, 1],
            [0, 1],
            [0, 1],
            # [0, 1],
            # [0, 1],
            # [0, 1],
            # [0, 1],
        ),
        "submetrics": True,
        "n_submetrics": 4,
        "submetric_names": [
            "first_pca_component",
            # "first_pca_component_velocity_alignment",
            "first_pca_component_horizontal_axis_alignment",
            # "first_pca_component_z_axis_alignment",
            "second_pca_component",
            "second_pca_component_velocity_alignment",
            # "second_pca_component_horizontal_axis_alignment",
            # "second_pca_component_z_axis_alignment",
        ],
    }
    order_metrics.append(PCA_velocity_metric)
    gaussian_entropy_metric = {
        "name": "gaussian_entropy",
        "function": gaussian_entropy,
        "values": [],
        "color": "darkred",
        "value_space": [-5, 5],
        "submetrics": False,
    }
    order_metrics.append(gaussian_entropy_metric)
    flight_deviation_metric = {
        "name": "flight_deviation_metric",
        "function": flight_deviations,
        "values": [],
        "color": "forestgreen",
        "value_space": [[0, 1.5], [0, 3], [0, 1.5]],
        "submetrics": True,
        "n_submetrics": 3,
        "submetric_names": [
            "parallel_deviation",
            "lateral_deviation",
            "vertical_deviation",
        ],
    }
    order_metrics.append(flight_deviation_metric)

    if i <= amount_of_analysises:
        print(f"Starting with analysis {i} / {amount_of_analysises}", flush=True)

    if filename.endswith(".trj"):
        filename = os.path.splitext(filename)[0]

        # launch analysis
        trajectory_analysis(filename, order_metrics, no_plotting=True)

    plt.close("all")
    gc.collect()
