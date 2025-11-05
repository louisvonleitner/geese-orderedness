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


def trajectory_analysis(filename: str, order_metrics: list):

    if len(order_metrics) != 4:
        raise Exception("More or less than 4 metrics given, but 4 expected!")
    # run metric computation
    order_metrics = calculate_metrics(order_metrics=order_metrics, filename=filename)

    # for data analysis return and do not plot
    # return order_metrics

    # edge case handling
    if order_metrics == None:
        print(f"Aborting process because no suitable trajectories were found.")
        return None

    print(f"Done!")

    """
    print(f"Computing Matrix Metrics of adjacency matrices...")
    # run matrix metric computation on adjacency matrices
    for metric in order_metrics:
        metric["algebraic_connectivities"] = calculate_algebraic_connectivity_over_time(
            metric
        )
    print(f"Done!")
    """

    for metric in order_metrics:
        if metric == None:
            return None

    print("Saving plots...")
    for metric in order_metrics:
        plot_distribution(
            metric,
            filename,
            showing=False,
            saving=True,
        )

    plot_metrics_over_time(order_metrics, filename, showing=False, saving=True)

    print("Done!")

    # ===================================================================================================
    # ALL DATA HAS BEEN ACQUIRED, NOW JUST ANIMATION PLOTTING
    # ===================================================================================================

    print(f"Generating animation...")
    animation(filename, order_metrics)

    print(f"Finished the whole process for file {filename}")
