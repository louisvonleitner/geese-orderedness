import numpy as np
import pandas as pd
from tqdm import trange
import os

from matplotlib import pyplot as plt

# project's own imports
from data_engineering.metric_calculation import (
    calculate_metrics,
)
from data_engineering.metric_evaluation import (
    calculate_algebraic_connectivity_over_time,
    plot_algebraic_connectivities,
    plot_distribution,
)
from visualization.animate_trajectory import animation


def trajectory_analysis(filename: str, metrics: dict):

    if len(metrics) != 2:
        raise Exception("More or less than 2 metrics given, but 2 expected!")
    # run metric computation
    metrics, entropies = calculate_metrics(metrics=metrics, filename=filename)

    # edge case handling
    if metrics == None:
        print(f"Aborting process because no suitable trajectories were found.")
        return None

    print(f"Done!")

    print(f"Computing Matrix Metrics of adjacency matrices...")
    # run matrix metric computation on adjacency matrices
    for metric in metrics:
        metric["algebraic_connectivities"] = calculate_algebraic_connectivity_over_time(
            metric
        )

        plot_distribution(metric, filename, showing=False, saving=True)

    print(f"Done!")

    print(f"Saving Matrix Metrics Graphs...")
    plot_algebraic_connectivities(metrics, filename, showing=False, saving=True)
    print(f"Done!")

    fig = plt.figure(figsize=(6, 6))

    plt.plot(entropies)
    plt.savefig(f"data/{filename}/figs/entropies.png")
    plt.close()

    # ===================================================================================================
    # ALL DATA HAS BEEN ACQUIRED, NOW JUST PLOTTING
    # ===================================================================================================

    print(f"Generating animation...")
    animation(filename, metrics)

    print(f"Finished the whole process for file {filename}")
