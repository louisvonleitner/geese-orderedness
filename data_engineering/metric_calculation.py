import pandas as pd
import numpy as np
import csv
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
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

    if type(locations) == list:
        return []

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


def calculate_velocity_alignment(geese: dict) -> float:
    """Calculate the normalized velocity alignment of a set of geese
    return normed_velocity_alignment"""
    normed_velocities = np.array(
        [
            geese[trj_id]["velocity"] / geese[trj_id]["velocity_norm"]
            for trj_id in geese
            if geese[trj_id]["velocity_norm"] != 0
        ]
    )

    if len(normed_velocities) == 0:
        return np.nan

    normed_velocity_alignment = np.linalg.norm(np.mean(normed_velocities, axis=0))

    return normed_velocity_alignment


def calculate_velocity_deviation(geese: dict) -> float:
    """Calculate the velocity alignment of a set of geese
    return velocity_alignment"""
    velocities = np.array(
        [
            geese[trj_id]["velocity"]
            for trj_id in geese
            if geese[trj_id]["velocity_norm"] != 0
        ]
    )

    if len(velocities) == 0:
        return np.nan

    velocity_deviation = np.linalg.norm(np.std(velocities, axis=0))

    return velocity_deviation


def calculate_velocity_PCA(geese: dict) -> tuple:
    """Calculate the first and second principal component of the velocity vectors"""

    def horizontal_perpendicular(v):
        vx, vy, vz = v
        if vx == 0 and vy == 0:
            # v is vertical → pick any horizontal direction
            return np.array([1.0, 0.0, 0.0])
        vector = np.array([-vy, vx, 0.0])
        normed_vector = vector / np.linalg.norm(vector)
        return normed_vector

    if len(geese) == 0:
        return tuple([np.nan for _ in range(10)])

    velocities = np.array(
        [
            geese[trj_id]["velocity"]
            for trj_id in geese
            if geese[trj_id]["velocity_norm"] != 0
        ]
    )
    average_velocity = np.mean(velocities, axis=0)
    average_velocity_normed = average_velocity / np.linalg.norm(average_velocity)

    # handling exceptions
    if velocities.size == 0 or len(velocities.shape) != 2 or velocities.shape[0] < 2:
        return tupe([np.nan for _ in range(10)])

    # set up PCA
    pca = PCA(n_components=2)

    # execute PCA on Data
    pca.fit(velocities)

    # extract PCA components
    first_component, second_component = pca.components_

    first_component_length = np.linalg.norm(first_component)
    second_component_length = np.linalg.norm(second_component)

    normed_first_component = first_component / first_component_length
    normed_second_component = second_component / second_component_length

    first_component_variance, second_component_variance = pca.explained_variance_

    # calculating alignment with velocity

    first_component_alignment = np.abs(
        np.dot(normed_first_component, average_velocity_normed)
    )
    second_component_alignment = np.abs(
        np.dot(normed_second_component, average_velocity_normed)
    )

    z_vector = np.array([0, 0, 1])
    # calculate z-axis alignment
    first_component_z_alignment = np.abs(np.dot(normed_first_component, z_vector))
    second_component_z_alignment = np.abs(np.dot(normed_second_component, z_vector))

    xy_plane_perp_vector = horizontal_perpendicular(average_velocity_normed)
    # calculate alignment with vector in xy plane perpendicular to flight direction
    first_component_xy_alignment = np.abs(
        np.dot(normed_second_component, xy_plane_perp_vector)
    )
    second_component_xy_alignment = np.abs(
        np.dot(normed_second_component, xy_plane_perp_vector)
    )

    return (
        first_component_variance,
        second_component_variance,
        first_component_alignment,
        second_component_alignment,
        first_component_xy_alignment,
        first_component_z_alignment,
        second_component_xy_alignment,
        second_component_z_alignment,
    )


def calculate_longitudinal_acceleration_deviation(geese: dict) -> float:
    """Calculate the longitudinal acceleration deviation of a set of geese
    return (
    xi_acceleration_deviation,
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

    if len(xi_accelerations) == 0:
        return np.nan

    # calculate standard deviation from mean acceleration in xi direction
    xi_acceleration_deviation = np.linalg.norm(np.std(xi_accelerations))
    return xi_acceleration_deviation


def calculate_sidewise_acceleration_deviation(geese: dict) -> float:
    """Calculate the sidewise acceleration deviation of a set of geese
    return combined sidewise acceleration deviation
    """

    # goose['raw_acceleration'] = {xi, eta, zeta}
    eta_accelerations = np.array(
        [
            geese[trj_id]["raw_acceleration"]["eta"]
            for trj_id in geese
            if geese[trj_id]["raw_acceleration"]["eta"] != 0
        ]
    )

    if len(eta_accelerations) == 0:
        return np.nan

    zeta_accelerations = np.array(
        [
            geese[trj_id]["raw_acceleration"]["zeta"]
            for trj_id in geese
            if geese[trj_id]["raw_acceleration"]["zeta"] != 0
        ]
    )

    if len(zeta_accelerations) == 0:
        return np.nan

    # calculate standard deviation from mean acceleration along each axis
    eta_acceleration_deviation = np.linalg.norm(np.std(eta_accelerations))
    zeta_acceleration_deviation = np.linalg.norm(np.std(zeta_accelerations))

    combined_sidewise_acceleration_deviation = (
        eta_acceleration_deviation + zeta_acceleration_deviation
    )

    return combined_sidewise_acceleration_deviation


def save_metric_output(order_metrics: list, average_directions, filename: str):

    functions = {}
    values = {}

    # make sure folder exists
    for metric in order_metrics:
        os.makedirs(
            os.path.dirname(f"data/{filename}/{metric["name"]}_values.csv"),
            exist_ok=True,
        )

        # keep track of metrics to reassign them later
        functions[metric["name"]] = metric["function"]
        values[metric["name"]] = metric["values"]

        # write values into csv file
        with open(
            f"data/{filename}/{metric["name"]}_values.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(metric["values"])

        # make empty to save easier
        metric["values"] = []
        metric["function"] = None

    with open(
        f"data/{filename}/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(order_metrics, f, ensure_ascii=False, indent=4)

    # save numpy array of average velocity vectors
    average_directions = [
        i for i in average_directions if type(i) == np.ndarray and i.shape[0] == 3
    ]
    average_directions = np.array(average_directions)
    np.save(f"data/{filename}/average_velocity_vectors.npy", average_directions)

    # reassign metrics
    for metric in order_metrics:
        metric["function"] = functions[metric["name"]]
        metric["values"] = values[metric["name"]]


def calculate_metrics(order_metrics: list, filename: str):
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

    print(f"Cleaning data...", flush=True)
    df, individual_geese_trjs, n_trjs = clean_data(df, individual_geese_trjs)

    # abort process
    if n_trjs == 0:
        print(f"No suitable trajectories!", flush=True)
        return None

    print(f"Data Cleaned!", flush=True)

    print(f"Computing metrics between birds...", flush=True)

    # defining loop length as the middle 75%
    first_frame = int(df["frame"].min())
    last_frame = int(df["frame"].max())
    length = last_frame - first_frame

    buffer_length = length * 0.125

    average_directions = []

    first_frame += buffer_length
    last_frame -= buffer_length

    first_frame = int(first_frame)
    last_frame = int(last_frame)

    # starting loop
    for frame in range(first_frame, last_frame):

        # get geese in nice dict
        geese = get_frame_geese(frame, individual_geese_trjs, column_names)

        # compute mean velocity vector (for crosswind determination)
        velocities = np.array([geese[trj_id]["velocity"] for trj_id in geese])

        if velocities.size > 0:
            velocity_mean = np.mean(velocities, axis=0)
            average_directions.append(velocity_mean)

        if geese == []:
            for metric in order_metrics:
                if metric["submetrics"] == True:
                    nan_tuple = tuple([np.nan] * metric["n_submetrics"])
                    metric["values"].append(nan_tuple)
                else:
                    metric["values"].append(np.nan)

        else:

            n_geese = len(geese)

            for metric in order_metrics:
                calculated_metric = metric["function"](geese)
                metric["values"].append(calculated_metric)

    print(f"Saving metrics...", flush=True)

    # save matrices in files
    save_metric_output(order_metrics, average_directions, filename)

    # end of metrics calculation
    return order_metrics
