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
)
from visualization.animate_trajectory import (
    animation,
    plot_distribution,
    plot_metrics_over_time,
)


def trajectory_analysis(filename: str, metrics: list):

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

    print(f"Done!")

    # print(f"Saving Matrix Metrics Graphs...")
    # plot_algebraic_connectivities(metrics, filename, showing=False, saving=True)
    # print(f"Done!")

    metrics = [metrics[0], np.array(entropies)]

    # saving plots
    print("Saving plots...")
    plot_distribution(
        metrics[0]["matrices"],
        "boltzmann algebraic connectivity",
        filename,
        showing=False,
        saving=True,
    )
    plot_distribution(metrics[1], "entropy", filename, showing=False, saving=True)

    metrics[0] = metrics[0]["algebraic_connectivities"]
    plot_metrics_over_time(metrics, filename, showing=False, saving=True)

    print("Done!")

    # ===================================================================================================
    # ALL DATA HAS BEEN ACQUIRED, NOW JUST PLOTTING
    # ===================================================================================================

    print(f"Generating animation...")
    animation(filename, metrics, entropies)

    print(f"Finished the whole process for file {filename}")
