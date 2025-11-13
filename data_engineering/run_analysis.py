import numpy as np
import pandas as pd
from tqdm import trange
import os

from matplotlib import pyplot as plt

# project's own imports
from data_engineering.metric_calculation import (
    calculate_metrics,
)

from visualization.animate_trajectory import (
    animation,
    plot_distribution,
    plot_metrics_over_time,
)


def extract_submetrics(metrics):

    submetrics = []
    for metric in metrics:
        if metric == None:
            return None

        if metric["submetrics"] == True:
            for i in range(metric["n_submetrics"]):
                submetric = metric.copy()

                # check for errors
                for value in metric["values"]:
                    if type(value) != tuple:
                        print(
                            f"Value not tuple with value: {value} for metric {metric['name']}"
                        )

                values = [j[i] for j in metric["values"]]

                submetric["name"] = metric["name"] + f"_submetric_{i + 1}"
                submetric["value_space"] = metric["value_space"][i]
                submetric["values"] = values
                submetrics.append(submetric)

            metrics.remove(metric)

    metrics = metrics + submetrics

    return metrics


def trajectory_analysis(filename: str, order_metrics: list, no_plotting=False):

    # run metric computation
    order_metrics = calculate_metrics(order_metrics=order_metrics, filename=filename)

    # for data analysis return and do not plot
    # return order_metrics

    # edge case handling
    if order_metrics == None:
        print(
            f"Aborting process because no suitable trajectories were found.", flush=True
        )
        return None

    print(f"Done!", flush=True)

    print("Extracting submetrics...", flush=True)
    order_metrics = extract_submetrics(order_metrics)

    if no_plotting == False:
        order_metrics = [
            m
            for m in order_metrics
            if len(m.get("values", [])) > 0 and not np.all(np.isnan(m["values"]))
        ]

        print("Saving plots...", flush=True)
        for metric in order_metrics:
            plot_distribution(
                metric,
                filename,
                showing=False,
                saving=True,
            )

        plot_metrics_over_time(order_metrics, filename, showing=False, saving=True)

        print("Done!", flush=True)

    # ===================================================================================================
    # ALL DATA HAS BEEN ACQUIRED, NOW JUST ANIMATION PLOTTING
    # ===================================================================================================

    if no_plotting == False:
        print(f"Generating animation...", flush=True)
        animation(filename, order_metrics)

    print(f"Finished the whole process for file {filename}", flush=True)

    return order_metrics
