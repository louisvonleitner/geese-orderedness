import pandas as pd
import numpy as np
from tqdm import trange
import csv
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns


def read_trajectory_data(filename, column_numbers, column_names):
    """Read data from file into a pandas dataframe and return this dataframe"""

    # dataframe including all trajectories
    df = pd.read_csv(
        filename,
        sep="\s+",
        usecols=column_numbers,
        names=column_names,
        dtype=np.float32,
    )

    df["trj_id"] = df["trj_id"].astype(int)

    # get number of trajectories
    n_trjs = int(df["trj_id"].max()) + 1

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

    if type(locations) == list:
        location_plotter = ax.scatter([], [], [], color="red")
    else:
        # iterate through geese and collect them in a dict
        for index, data in locations.iterrows():
            trj_id, xpos, ypos, zpos, xvel, yvel, zvel, xi, eta, zeta = data[
                column_names
            ]

            # structured data
            trj_id = int(trj_id)
            position = np.array([xpos, ypos, zpos])
            velocity = np.array([xvel, yvel, zvel])

            # compute cartesian acceleration from directed accelerations
            acceleration = acceleration_cartesian(velocity, xi, eta, zeta)

            # store data in dict
            goose = {
                "trj_id": trj_id,
                "position": position,
                "velocity": velocity,
                "velocity_norm": np.linalg.norm(velocity),
                "acceleration": acceleration,
                "acceleration_norm": np.linalg.norm(acceleration),
            }

            # save in geese dict
            geese[trj_id] = goose

    return geese


def sigmoid(x):
    return 1 / (1 + np.exp(x))


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


def boltzmann_metric(
    goose_1: dict,
    goose_2: dict,
) -> float:

    beta = 0.5
    a = 1
    b = 1
    c = 1

    if (
        goose_1["velocity_norm"] == 0
        or goose_2["velocity_norm"] == 0
        or goose_1["acceleration_norm"] == 0
        or goose_2["acceleration_norm"] == 0
    ):
        # cannot compute boltzmann weight -> set to 0
        return 0

    # alignments
    velocity_alignment = np.dot(goose_1["velocity"], goose_2["velocity"])
    acceleration_alignment = np.dot(goose_1["acceleration"], goose_2["acceleration"])

    # norms
    velocity_norm = goose_1["velocity_norm"] * goose_2["velocity_norm"]
    acceleration_norm = goose_2["acceleration_norm"] * goose_2["acceleration_norm"]

    # factors
    distance_factor = sum((goose_1["position"] - goose_2["position"]) ** 2)
    velocity_factor = velocity_alignment / velocity_norm
    acceleration_factor = acceleration_alignment / acceleration_norm

    # boltzmann compute
    H = a * distance_factor - b * velocity_factor - c * acceleration_factor

    boltzmann_weight = np.exp(-(beta * H))

    return boltzmann_weight


def distance_metric(
    goose_1: dict,
    goose_2: dict,
) -> float:

    distance = np.linalg.norm(goose_1["position"] - goose_2["position"])
    return distance


def save_metric_output(metrics: list, filename: str):
    # make sure folder exists
    for metric in metrics:
        os.makedirs(
            os.path.dirname(
                f"C:/Python Projects/tohoku_university/geese_project/data/{filename}/{metric["name"]}_matrices.csv"
            ),
            exist_ok=True,
        )

        with open(
            f"C:/Python Projects/tohoku_university/geese_project/data/{filename}/{metric["name"]}_matrices.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            for i, matrix in enumerate(metric["matrices"]):
                writer.writerows(matrix)
                writer.writerow([])

    # save metrics used
    for metric in metrics:
        metric["matrices"] = []
        metric["function"] = []

    with open(
        f"C:/Python Projects/tohoku_university/geese_project/data/{filename}/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


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
    filepath = f"C:/Python Projects/tohoku_university/geese_project/data/{filename}.trj"
    # =====================================================================================================

    # the column names of data being used
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
        filepath, file_column_numbers, file_column_names
    )

    # defining loop length
    first_frame = int(df["frame"].min())
    last_frame = int(df["frame"].max())

    # starting loop
    for frame in trange(first_frame, last_frame):

        # get geese in nice dict
        geese = get_frame_geese(frame, individual_geese_trjs, column_names)

        n_geese = len(geese)

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
    save_metric_output(metrics, filename)

    print(f"Done!")

    # end of metrics calculation
    return metrics


"""
# define metrics
metrics = []

# distance metric
distance_metric_dict = {
    "name": "distance",
    "function": distance_metric,
    "matrices": [],
    "symmetric": True,
    "color": "blue",
}
metrics.append(distance_metric_dict)

# clusteriness metric
clusteriness_metric_dict = {
    "name": "clusteriness",
    "function": clusteriness_metric,
    "matrices": [],
    "symmetric": True,
    "color": "yellow",
}
metrics.append(clusteriness_metric_dict)

# boltzmann metric
boltzmann_metric_dict = {
    "name": "boltzmann",
    "function": boltzmann_metric,
    "matrices": [],
    "symmetric": True,
    "color": "green",
}
metrics.append(boltzmann_metric_dict)
"""


# define metrics
metrics = []


param_list = np.linspace(0.01, 2, 10)
b_list = np.linspace(0.2, 10, 10)
c_list = np.linspace(0.2, 10, 10)
beta_list = np.linspace(0.1, 1, 10)


for a in param_list:

    b = 1
    c = 1
    beta = 0.5

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
        velocity_alignment = np.dot(goose_1["velocity"], goose_2["velocity"])
        acceleration_alignment = np.dot(
            goose_1["acceleration"], goose_2["acceleration"]
        )

        # norms
        velocity_norm = goose_1["velocity_norm"] * goose_2["velocity_norm"]
        acceleration_norm = goose_2["acceleration_norm"] * goose_2["acceleration_norm"]

        # factors
        distance_factor = sum((goose_1["position"] - goose_2["position"]) ** 2)
        velocity_factor = velocity_alignment / velocity_norm
        acceleration_factor = acceleration_alignment / acceleration_norm

        # boltzmann compute
        H = a * distance_factor - b * velocity_factor - c * acceleration_factor

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


calculate_metrics(metrics=metrics, filename="20201206-S6F1820E1#3S20")
