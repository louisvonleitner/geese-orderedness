import numpy as np
import pandas as pd
from tqdm import trange

import matplotlib.pyplot as plt

# project's own imports
from data_engineering.metric_calculation import (
    calculate_metrics,
    boltzmann_metric,
    inverse_exponential_distance_metric,
)
from data_engineering.graph_theory import (
    compute_algebraic_connectivity,
)
from data_engineering.metric_evaluation import (
    load_metrics,
    calculate_algebraic_connectivity_over_time,
)

# define metrics
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


# ===================================================================================================
filename = "20201206-S6F1820E1#3S20"
# ===================================================================================================

# run metric computation
metrics = calculate_metrics(metrics=metrics, filename=filename)

# run matrix metric computation on adjacency matrices
for metric in metrics:
    calculate_algebraic_connectivity_over_time(metric)

# ===================================================================================================
# ALL DATA HAS BEEN ACQUIRED, NOW JUST PLOTTING
# ===================================================================================================
