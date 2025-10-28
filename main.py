import numpy as np
import pandas as pd
from tqdm import trange
import os

# project's own imports
from data_engineering.run_analysis import trajectory_analysis
from data_engineering.metric_calculation import (
    boltzmann_metric,
    inverse_exponential_distance_metric,
)


# ===================================================================================================
folder_path = "data/trajectory_data"
# ===================================================================================================

amount_of_analysises = len(os.listdir(folder_path)) - 1

i = 0
for filename in os.listdir(folder_path):
    i += 1
    metrics = []

    boltzmann_metric_dict = {
        "name": f"boltzmann",
        "function": boltzmann_metric,
        "matrices": [],
        "symmetric": True,
        "color": "green",
    }
    metrics.append(boltzmann_metric_dict)

    inverse_exponential_distance_metric_dict = {
        "name": f"inverse_exponential_distance",
        "function": inverse_exponential_distance_metric,
        "matrices": [],
        "symmetric": True,
        "color": "green",
    }
    metrics.append(inverse_exponential_distance_metric_dict)

    print(f"Starting with analysis {i} / {amount_of_analysises}")

    if filename.endswith(".trj"):
        filename = os.path.splitext(filename)[0]

        # launch analysis
        trajectory_analysis(filename, metrics)
