import pandas as pd
import numpy as np
from tqdm import trange
import csv
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns

from data_engineering.clean_trajectory import clean_data


def read_trajectory_data(filename, column_numbers, column_names):
    """Read data from file into a pandas dataframe and return this dataframe"""

    # dataframe including all trajectories
    df = pd.read_csv(
        f"data/trajectory_data/{filename}.trj",
        sep="\s+",
        usecols=column_numbers,
        names=column_names,
        dtype=np.float32,
    )

    df["trj_id"] = df["trj_id"].astype(int)

    # get number of trajectories
    n_trjs = len(df["trj_id"].unique())

    # list of dataframes, where one dataframe holds exactly one trajectory
    individual_geese_trjs = [group_df for trj_id, group_df in df.groupby("trj_id")]

    return df, individual_geese_trjs, n_trjs


def get_frame_locations(frame: int, individual_geese_trjs: list, column_names: list):
    """return locations of all geese currently visible by the camera for a specific frame
    returns location data as a pandas Dataframe  xloc, yloc, zloc"""

    locations = []
    for trj in individual_geese_trjs:
        # grab location data based on frame
        location_data = trj[trj["frame"] == frame]

        if not location_data.empty:
            locations.append(location_data[column_names])

    if locations != []:
        locations = pd.concat(locations)

    return locations


def get_frame_geese(
    frame: int, individual_geese_trjs: list, column_names: list
) -> dict:
    """get a dictionary of geese with position, velocity and acceleration information indexed by trj_id"""

    # update locations of geese and plot them
    locations = get_frame_locations(frame, individual_geese_trjs, column_names)

    # array of geese indexed by trj_id
    geese = {}

    # iterate through geese and collect them in a dict
    for index, data in locations.iterrows():
        trj_id, xpos, ypos, zpos, xvel, yvel, zvel, xi, eta, zeta = data[column_names]

        # structured data
        trj_id = int(trj_id)
        position = np.array([xpos, ypos, zpos])
        velocity = np.array([xvel, yvel, zvel])

        # compute cartesian acceleration from directed accelerations
        raw_acceleration = {"xi": xi, "eta": eta, "zeta": zeta}
        acceleration = acceleration_cartesian(velocity, xi, eta, zeta)

        # store data in dict
        goose = {
            "trj_id": trj_id,
            "position": position,
            "velocity": velocity,
            "velocity_norm": np.linalg.norm(velocity),
            "acceleration": acceleration,
            "acceleration_norm": np.linalg.norm(acceleration),
            "raw_acceleration": raw_acceleration,
        }

        # save in geese dict
        geese[trj_id] = goose

    return geese


def acceleration_cartesian(
    v: np.ndarray, xi: float, eta: float, zeta: float
) -> np.ndarray:
    """
    Compute the Cartesian acceleration vector given components along
    an orthogonal (not necessarily orthonormal) basis defined by a direction vector v.

    Parameters
    ----------
    v : array-like, shape (3,)
        Reference direction vector.
    xi : float
        Acceleration along v.
    eta : float
        Acceleration along horizontal perpendicular axis (to the right when values are +).
    zeta : float
        Acceleration along perpendicular vertical axis (to the top when values are +).


    """

    v = np.array(v, dtype=float)
    v_norm = np.linalg.norm(v)

    # if vector is 0 vector say there is no acceleration
    if v_norm == 0:
        v_dir = np.array([0, 0, 0])
    else:
        v_dir = v / np.linalg.norm(v)

    # Define world vertical (z-axis)
    z_axis = np.array([0.0, 0.0, 1.0])

    # Compute horizontal perpendicular direction (eta_dir)
    eta_dir = np.cross(z_axis, v_dir)
    if np.linalg.norm(eta_dir) < 1e-8:
        # v is parallel to z-axis; pick arbitrary horizontal axis
        eta_dir = np.array([1.0, 0.0, 0.0])
    else:
        eta_dir = eta_dir / np.linalg.norm(eta_dir)

    # Compute the third orthogonal direction (zeta_dir)
    zeta_dir = np.cross(eta_dir, v_dir)

    # Combine the three components
    a_cartesian = xi * v_dir - eta * eta_dir - zeta * zeta_dir
    return a_cartesian


def clusteriness_metric(
    goose_1: dict,
    goose_2: dict,
) -> float:
    """calculate and return the distance, acceleration_clusteriness and boltzmann weight between two birds"""

    if (
        goose_1["velocity_norm"] == 0
        or goose_2["velocity_norm"] == 0
        or goose_1["acceleration_norm"] == 0
        or goose_2["acceleration_norm"] == 0
    ):
        # cannot compute clusteriness -> set to 0
        return 0

    # alignments
    velocity_alignment = np.dot(goose_1["velocity"], goose_2["velocity"])
    acceleration_alignment = np.dot(goose_1["acceleration"], goose_2["acceleration"])

    # norms
    velocity_norm = goose_1["velocity_norm"] * goose_2["velocity_norm"]
    acceleration_norm = goose_1["acceleration_norm"] * goose_2["acceleration_norm"]

    # factors
    distance_factor = 1 / np.linalg.norm(goose_1["position"] - goose_2["position"])
    velocity_factor = velocity_alignment / velocity_norm
    acceleration_factor = acceleration_alignment / acceleration_norm

    # final compute
    clusteriness = distance_factor * velocity_factor * acceleration_factor

    return clusteriness


def distance_metric(
    goose_1: dict,
    goose_2: dict,
) -> float:

    distance = np.linalg.norm(goose_1["position"] - goose_2["position"])
    return distance


def calculate_entropy(geese: dict) -> float:

    # read from distance_distribution data chart
    # might need different normalization method
    max_normal_distance = 180

    # max_normal_distance_spread = 21  # 40 seemed viable before

    n = len(geese)

    if n < 2:
        return np.nan

    positional_weight = 1
    # distance_spread_weight = 1
    velocity_weight = 1
    acceleration_weight = 1

    geese_positions = [geese[trj_id]["position"] for trj_id in geese]
    normed_velocities = [
        geese[trj_id]["velocity"] / geese[trj_id]["velocity_norm"]
        for trj_id in geese
        if geese[trj_id]["velocity_norm"] != 0
    ]

    normed_accelerations = [
        geese[trj_id]["acceleration"] / geese[trj_id]["acceleration_norm"]
        for trj_id in geese
        if geese[trj_id]["acceleration_norm"] != 0
    ]

    geese_positions = np.array(geese_positions)
    normed_velocities = np.array(normed_velocities)
    normed_accelerations = np.array(normed_accelerations)

    # center of locations
    mu = np.mean(geese_positions)

    distances = np.linalg.norm(geese_positions - mu, axis=1)

    # deviation from center
    positional_deviation = np.mean(distances)
    normed_positional_deviation = positional_deviation / max_normal_distance

    # deviation from average distance to center
    # distance_spread = np.std(distances)
    # normed_distance_spread = distance_spread / max_normal_distance_spread

    # velocity alignment
    vel_align = np.linalg.norm(np.mean(normed_velocities, axis=0))
    vel_spread = 1 - vel_align

    # acceleration alignment
    acc_align = np.linalg.norm(np.mean(normed_accelerations, axis=0))
    acc_spread = 1 - acc_align

    # compute normalized entropy
    entropy = (
        positional_weight * normed_positional_deviation
        # + distance_spread_weight * normed_distance_spread
        + velocity_weight * vel_spread
        + acceleration_weight * acc_spread
    ) / (
        positional_weight
        # + distance_spread_weight
        + velocity_weight
        + acceleration_weight
    )

    return entropy


def calculate_velocity_alignment(geese: dict) -> tuple:
    """Calculate the velocity alignment of a set of geese
    return (velocity_alignment, normed_velocity_alignment)"""
    velocities = np.array(
        [
            geese[trj_id]["velocity"]
            for trj_id in geese
            if geese[trj_id]["velocity_norm"] != 0
        ]
    )

    velocity_lengths = np.array(
        [
            geese[trj_id]["velocity_norm"]
            for trj_id in geese
            if geese[trj_id]["velocity_norm"] != 0
        ]
    )

    # TODO: Use normed velocities or full value velocities?
    # TODO: Use velocity deviation as a metric aswell?
    normed_velocities = np.array(
        [
            geese[trj_id]["velocity"] / geese[trj_id]["velocity_norm"]
            for trj_id in geese
            if geese[trj_id]["velocity_norm"] != 0
        ]
    )

    velocity_alignment = np.linalg.norm(np.mean(velocities, axis=0))
    velocity_alignment = velocity_alignment / np.mean(velocity_lengths)
    normed_velocity_alignment = np.linalg.norm(np.mean(normed_velocities, axis=0))

    return velocity_alignment, normed_velocity_alignment


def calculate_acceleration_deviation(geese: dict):
    """Calculate the acceleration deviation of a set of geese
    return (
    xi_acceleration_deviation,
    eta_acceleration_deviation,
    zeta_acceleration_deviation,
    )
    """

    # goose['raw_acceleration'] = {xi, eta, zeta}

    xi_accelerations = np.array(
        [
            geese[trj_id]["raw_acceleration"]["xi"]
            for trj_id in geese
            if geese[trj_id]["raw_acceleration"]["xi"] != 0
        ]
    )

    eta_accelerations = np.array(
        [
            geese[trj_id]["raw_acceleration"]["eta"]
            for trj_id in geese
            if geese[trj_id]["raw_acceleration"]["eta"] != 0
        ]
    )

    zeta_accelerations = np.array(
        [
            geese[trj_id]["raw_acceleration"]["zeta"]
            for trj_id in geese
            if geese[trj_id]["raw_acceleration"]["zeta"] != 0
        ]
    )

    # calculate standard deviation from mean acceleration along each axis
    xi_acceleration_deviation = np.linalg.norm(np.std(xi_accelerations))
    eta_acceleration_deviation = np.linalg.norm(np.std(eta_accelerations))
    zeta_acceleration_deviation = np.linalg.norm(np.std(zeta_accelerations))

    return (
        xi_acceleration_deviation,
        eta_acceleration_deviation,
        zeta_acceleration_deviation,
    )


def save_metric_output(metrics: list, entropies: list, filename: str):
    # make sure folder exists
    for metric in metrics:
        os.makedirs(
            os.path.dirname(f"data/{filename}/{metric["name"]}_matrices.csv"),
            exist_ok=True,
        )

        with open(
            f"data/{filename}/{metric["name"]}_matrices.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            for i, matrix in enumerate(metric["matrices"]):
                writer.writerows(matrix)
                writer.writerow([])

    matrices = {}
    functions = {}

    # keep track of metrics to reassign them later
    for metric in metrics:
        matrices[metric["name"]] = metric["matrices"]
        functions[metric["name"]] = metric["function"]

        # make empty to save easier
        metric["matrices"] = []
        metric["function"] = []

    with open(
        f"data/{filename}/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # reassign metrics
    for metric in metrics:
        metric["matrices"] = matrices[metric["name"]]
        metric["function"] = functions[metric["name"]]

    # keep track of entropy
    os.makedirs(
        os.path.dirname(f"data/{filename}/{metric["name"]}_entropies.csv"),
        exist_ok=True,
    )

    with open(
        f"data/{filename}/{metric["name"]}_entropies.csv",
        "w",
        newline="",
    ) as f:
        json.dump(entropies, f, ensure_ascii=False, indent=4)


def calculate_metrics(metrics: list, filename: str):
    """Calculate the Metrics between birds for every frame in the trajectory data and return the lists of adjacency matrices"""

    # ====================================================================================================
    # setting values to be used from dataset
    file_column_numbers = [0, 1, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18]
    file_column_names = [
        "trj_id",
        "frame",
        "xpos",
        "ypos",
        "zpos",
        "xvel",
        "yvel",
        "zvel",
        "n",
        "xi",
        "eta",
        "zeta",
    ]
    # =====================================================================================================

    # the column names of data being used for actual metrics and plotting
    column_names = [
        "trj_id",
        "xpos",
        "ypos",
        "zpos",
        "xvel",
        "yvel",
        "zvel",
        "xi",
        "eta",
        "zeta",
    ]

    # creating dataframes from function
    df, individual_geese_trjs, n_trjs = read_trajectory_data(
        filename, file_column_numbers, file_column_names
    )

    print(f"Cleaning data...")
    df, individual_geese_trjs, n_trjs = clean_data(df, individual_geese_trjs)

    # abort process
    if n_trjs == 0:
        print(f"No suitable trajectories!")
        return None, []

    print(f"Data Cleaned!")

    print(f"Computing metrics between birds...")

    # defining loop length as the middle 75%
    first_frame = int(df["frame"].min())
    last_frame = int(df["frame"].max())
    length = last_frame - first_frame

    buffer_length = length * 0.125

    first_frame += buffer_length
    last_frame -= buffer_length

    first_frame = int(first_frame)
    last_frame = int(last_frame)

    # entropy list
    entropies = []

    # starting loop
    for frame in trange(first_frame, last_frame):

        # get geese in nice dict
        geese = get_frame_geese(frame, individual_geese_trjs, column_names)

        n_geese = len(geese)

        # track entropy
        entropy = calculate_entropy(geese)
        entropies.append(entropy)

        # create adjacency matrix place holders for every metric
        for metric in metrics:
            metric["matrices"].append(np.zeros((n_geese, n_geese)))

        # iterate through geesee and calculate metric matrices
        i = -1
        for first_goose_index in geese:

            i += 1
            goose_1 = geese[first_goose_index]

            j = -1
            for second_goose_index in geese:

                j += 1
                goose_2 = geese[second_goose_index]

                if goose_1["trj_id"] == goose_2["trj_id"]:
                    pass

                # if the symmetric value has been computed already
                elif metrics[0]["matrices"][-1][i][j] != 0:
                    pass

                # if adjacency_matrix[trj_id][other_trj_id] == 0:
                else:
                    for metric in metrics:

                        # do the calculation
                        metric_weight = metric["function"](goose_1, goose_2)

                        # setting weight in adjacency matrix
                        metric["matrices"][-1][i][j] = metric_weight

                        if metric["symmetric"] == True:
                            metric["matrices"][-1][j][i] = metric_weight

    print(f"Saving metrics...")

    # save matrices in files
    save_metric_output(metrics, entropy, filename)

    # end of metrics calculation
    return metrics, entropies


# define metrics
metrics = []


def boltzmann_metric(
    goose_1: dict,
    goose_2: dict,
) -> float:

    a = 2 * 1 / 2500
    b = 1
    c = 1
    beta = 1

    if (
        goose_1["velocity_norm"] == 0
        or goose_2["velocity_norm"] == 0
        or goose_1["acceleration_norm"] == 0
        or goose_2["acceleration_norm"] == 0
    ):
        # cannot compute boltzmann weight -> set to 0
        return 0.0

    # alignments
    velocity_alignment = np.dot(goose_1["velocity"], goose_2["velocity"])
    acceleration_alignment = np.dot(goose_1["acceleration"], goose_2["acceleration"])

    # norms
    velocity_norm = goose_1["velocity_norm"] * goose_2["velocity_norm"]
    acceleration_norm = goose_1["acceleration_norm"] * goose_2["acceleration_norm"]

    # factors
    distance_factor = sum((goose_1["position"] - goose_2["position"]) ** 2)
    velocity_factor = velocity_alignment / velocity_norm
    acceleration_factor = acceleration_alignment / acceleration_norm

    # boltzmann compute
    H = a * distance_factor - b * velocity_factor - c * acceleration_factor

    boltzmann_weight = np.exp(-(beta * H))

    return boltzmann_weight


def inverse_exponential_distance_metric(
    goose_1: dict,
    goose_2: dict,
    alpha=2,
) -> float:

    # inverse exponential scalar
    a = 1 / 1600

    # factor
    distance_factor = sum((goose_1["position"] - goose_2["position"]) ** alpha)

    if distance_factor == 0:
        raise Exception("Geese are in the same position!")

    # inverse exponential compute
    H = a * distance_factor
    weight = np.exp(-H)

    return weight


"""
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
"""
