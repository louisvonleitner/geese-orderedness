import pandas as pd
import numpy as np
from tqdm import trange
import csv
import json
import os

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# project's own modules


def calculate_laplacian_matrix(matrix):
    """Using formula L = D - A to compute laplacian matrix"""

    # calculate (TODO in or out?) degrees
    degrees = matrix.sum(axis=1)

    # create degree matrix
    degree_matrix = np.diag(degrees)

    laplacian = degree_matrix - matrix

    return laplacian, degrees


def compute_algebraic_connectivity(matrix: np.ndarray, tol=1e-12) -> float:
    """compute laplaican matrix and its eigenvalues"""

    n = matrix.shape[0]

    laplacian, degrees = calculate_laplacian_matrix(matrix)

    laplacian = np.nan_to_num(laplacian, nan=0.0, posinf=0.0, neginf=0.0)

    # Initialize D^{-1/2}
    D_inv_sqrt = np.zeros((n, n))
    for i, d in enumerate(degrees):
        if d > 0:
            D_inv_sqrt[i, i] = 1.0 / np.sqrt(d)
        else:
            D_inv_sqrt[i, i] = 0.0  # isolated node

    normalized_laplacian = np.eye(n) - D_inv_sqrt @ matrix @ D_inv_sqrt

    normalized_laplacian = np.nan_to_num(
        normalized_laplacian, nan=0.0, posinf=0.0, neginf=0.0
    )

    laplacian_eigenvalues_normed = np.linalg.eigvals(normalized_laplacian)
    eigenvalues_sorted = np.sort(laplacian_eigenvalues_normed)

    if n <= 2:
        return eigenvalues_sorted[0]

    algebraic_connectivity = eigenvalues_sorted[1]

    return algebraic_connectivity


def read_frames_from_csv(filename):
    frames = []
    current_frame = []

    with open(f"data/trajectory_data/{filename}.trj", "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # Blank row signals new frame
                if current_frame:
                    frames.append(np.array(current_frame, dtype=float))
                    current_frame = []
            else:
                current_frame.append([float(x) for x in row])

        # Add the last frame if file didn't end with a blank line
        if current_frame:
            frames.append(np.array(current_frame, dtype=float))

    return frames


def read_metric_matrices(metrics: list, filename: str):

    for metric in metrics:
        matrices = read_frames_from_csv(filename)
        metric["matrices"] = matrices

    return metrics


def plot_distribution(metric: dict, filename: str, showing=True, saving=False):

    fig = plt.figure(figsize=(8, 5))
    all_values = np.concatenate([m.ravel() for m in metric["matrices"]])
    # ignoring 0 values
    all_values = all_values[all_values > 0]
    all_values = all_values[np.isfinite(all_values)]
    # threshold = np.percentile(all_values, 80)
    # all_values = all_values[all_values <= threshold]

    counts, bins = np.histogram(all_values, bins=100)

    sns.lineplot(x=bins[:-1], y=counts, color=metric["color"])

    plt.grid(color="lightgrey")

    # figure prettiness
    plt.title(f"{metric["name"]} Distribution")
    plt.xlabel(f"{metric["name"]}")

    if showing == True:
        plt.show()

    if saving == True:
        os.makedirs(
            os.path.dirname(f"data/{filename}/figs/{metric["name"]}_distribution.png"),
            exist_ok=True,
        )
        plt.savefig(f"data/{filename}/figs/{metric['name']}_distribution.png")
        plt.close()


def calculate_algebraic_connectivity_over_time(metric: dict):
    algebraic_connectivities = []
    for matrix_index in range(len(metric["matrices"])):
        matrix = metric["matrices"][matrix_index]

        algebraic_connectivity = compute_algebraic_connectivity(matrix)

        algebraic_connectivities.append(algebraic_connectivity)

    metric["algebraic_connectivities"] = algebraic_connectivities

    return algebraic_connectivities


def plot_algebraic_connectivities(
    metrics: list, filename: str, showing=True, saving=False
):

    fig, ax = plt.subplots(1, len(metrics), figsize=(10, 4))

    frames = [i for i in range(len(metrics[0]["matrices"]))]

    data_points = ["boltzmann", "inverse exponential distance"]

    for j in range(len(metrics)):
        metric = metrics[j]
        plot_axis = ax[j]
        sns.lineplot(
            x=np.array(frames),
            y=np.array(metric["algebraic_connectivities"]),
            ax=plot_axis,
            color=metric["color"],
        )

        plot_axis.set_title(f"""{metric['name']} over time""")
        plot_axis.set_xlabel(f"""frame""")
        plot_axis.set_ylabel(f"""{metric['name']}""")
        plot_axis.grid(color="lightgrey")

    plt.tight_layout()

    if showing == True:
        plt.show()

    if saving == True:
        os.makedirs(
            os.path.dirname(
                f"data/{filename}/figs/{metric["name"]}_algebraic_connectivity.png"
            ),
            exist_ok=True,
        )
        plt.savefig(f"data/{filename}/figs/{metric['name']}_algebraic_connectivity.png")
        plt.close()


def load_metrics(filename):
    with open(
        f"data/{filename}/metrics.json",
        "r",
        encoding="utf-8",
    ) as f:
        metrics = json.load(f)

    return metrics
