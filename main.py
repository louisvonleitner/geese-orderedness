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
    calculate_velocity_deviation,
    calculate_velocity_alignment,
    calculate_longitudinal_acceleration_deviation,
    calculate_sidewise_acceleration_deviation,
    calculate_velocity_PCA,
)


# ===================================================================================================
folder_path = "data/trajectory_data"
# ===================================================================================================

amount_of_analysises = len(os.listdir(folder_path)) - 1

i = 0
for filename in os.listdir(folder_path):
    i += 1

    order_metrics = []

    normalized_velocity_alignment_metric = {
        "name": "normalized_velocity_alignment",
        "function": calculate_velocity_alignment,
        "values": [],
        "color": "forestgreen",
        "value_space": [0.8, 1],
        "submetrics": False,
    }
    order_metrics.append(normalized_velocity_alignment_metric)
    velocity_deviation_metric = {
        "name": "velocity_deviation",
        "function": calculate_velocity_deviation,
        "values": [],
        "color": "green",
        "value_space": [0, 2.25],
        "submetrics": False,
    }
    order_metrics.append(velocity_deviation_metric)
    longitudinal_acceleration_metric = {
        "name": "longitudinal_acceleration_deviation",
        "function": calculate_longitudinal_acceleration_deviation,
        "values": [],
        "color": "red",
        "value_space": [0, 1.25],
        "submetrics": False,
    }
    order_metrics.append(longitudinal_acceleration_metric)
    sidewise_acceleration_metric = {
        "name": "sidewise_acceleration_deviation",
        "function": calculate_sidewise_acceleration_deviation,
        "values": [],
        "color": "crimson",
        "value_space": [0, 4],
        "submetrics": False,
    }
    order_metrics.append(sidewise_acceleration_metric)
    # PCA velocity metrics
    PCA_velocity_metric = {
        "name": "PCA_velocity_metric",
        "function": calculate_velocity_PCA,
        "values": [],
        "color": "blue",
        "value_space": ([0, 10], [0, 1]),
        "submetrics": True,
        "n_submetrics": 2,
    }
    order_metrics.append(PCA_velocity_metric)

    if i <= amount_of_analysises:
        print(f"Starting with analysis {i} / {amount_of_analysises}", flush=True)

    if filename.endswith(".trj"):
        filename = os.path.splitext(filename)[0]

        # launch analysis
        trajectory_analysis(filename, order_metrics)

    plt.close("all")
    gc.collect()
