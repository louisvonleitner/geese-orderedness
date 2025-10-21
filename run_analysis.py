import numpy as np
import pandas as pd
from tqdm import trange

import matplotlib.pyplot as plt

# project's own imports
from data_engineering.metric_calculation import (
    calculate_metrics,
    distance_metric,
    boltzmann_metric,
)
from data_engineering.graph_theory import (
    compute_algebraic_connectivity_and_connected_components,
)

filename = "20201206-S6F1820E1#3S20"

# define metrics
metrics = []


a_list = np.linspace(0.2, 2, 5)
b_list = np.linspace(0.2, 10, 5)
c_list = np.linspace(0.2, 10, 5)
beta_list = np.linspace(0.1, 1, 5)

for beta in beta_list:
    for a in a_list:
        for b in b_list:
            for c in c_lsit:

                def boltzmann_metric(
                    goose_1: dict,
                    goose_2: dict,
                ) -> float:

                    if (
                        goose_1["velocity_norm"] == 0
                        or goose_2["velocity_norm"] == 0
                        or goose_1["acceleration_norm"] == 0
                        or goose_2["acceleration_norm"] == 0
                    ):
                        # cannot compute boltzmann weight -> set to 0
                        return 0

                    # alignments
                    velocity_alignment = np.dot(
                        goose_1["velocity"], goose_2["velocity"]
                    )
                    acceleration_alignment = np.dot(
                        goose_1["acceleration"], goose_2["acceleration"]
                    )

                    # norms
                    velocity_norm = goose_1["velocity_norm"] * goose_2["velocity_norm"]
                    acceleration_norm = (
                        goose_2["acceleration_norm"] * goose_2["acceleration_norm"]
                    )

                    # factors
                    distance_factor = sum(
                        (goose_1["position"] - goose_2["position"]) ** 2
                    )
                    velocity_factor = velocity_alignment / velocity_norm
                    acceleration_factor = acceleration_alignment / acceleration_norm

                    # boltzmann compute
                    H = (
                        a * distance_factor
                        - b * velocity_factor
                        - c * acceleration_factor
                    )

                    boltzmann_weight = np.exp(-(beta * H))

                    return boltzmann_weight

                metric = {
                    "name": f"boltzmann a={a}, b={b}, c={c}, beta={beta}",
                    "function": boltzmann_metric,
                    "matrices": [],
                    "symmetric": True,
                    "color": "green",
                }
                metrics.append(metric)
