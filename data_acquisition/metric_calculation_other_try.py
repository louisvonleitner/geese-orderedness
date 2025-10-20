import pandas as pd
import numpy as np
from tqdm import trange
import csv

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


def get_frame_locations(frame, individual_geese_trjs, column_names):
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


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def acceleration_cartesian(v, xi, eta, zeta):
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

    Returns
    -------
    a_cartesian : np.ndarray, shape (3,)
        Cartesian acceleration vector.
    basis : tuple of np.ndarray
        The three orthogonal basis vectors (v_dir, eta_dir, zeta_dir).
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
    return a_cartesian, (v_dir, eta_dir, zeta_dir)


def calculate_clusteriness_and_distance(pos_1, pos_2, vel_1, vel_2, accel_1, accel_2):
    """calculate and return the clusteriness, acceleration_clusteriness, distance and asymetric_metric between two birds"""
    distance = np.linalg.norm(pos_1 - pos_2)
    dot_prod = np.dot(vel_1, vel_2)

    norming_factor = np.linalg.norm(vel_1) * np.linalg.norm(vel_2)
    if norming_factor != 0:
        clusteriness = (1 / distance) * (dot_prod / norming_factor)

        acceleration_alignment = np.dot(accel_1, accel_2)
        acceleration_norm = np.linalg.norm(accel_1) * np.linalg.norm(accel_2)

        # avoid division by 0
        if acceleration_norm != 0:
            acceleration_clusteriness = (
                clusteriness * acceleration_alignment / acceleration_norm
            )

        else:
            acceleration_clusteriness = 0

        # if other bird in front, asymmetric metric
        if np.dot((pos_1 - pos_2), (vel_1 + vel_2)) < 0:
            asymetric_metric = acceleration_clusteriness
        else:
            asymetric_metric = 0

        clusteriness = sigmoid(clusteriness)
        acceleration_clusteriness = sigmoid(acceleration_clusteriness)
        asymetric_metric = sigmoid(asymetric_metric)

    # division by 0 avoided (unclear speed = 0 !)
    else:
        clusteriness = 0
        acceleration_clusteriness = 0
        asymetric_metric = 0

    return clusteriness, acceleration_clusteriness, distance, asymetric_metric


def calculate_min_distances(individual_geese_trjs, first_frame, last_frame, n_trjs):

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

    distance_matrices = []
    clusteriness_matrices = []
    acceleration_clusteriness_matrices = []
    asymetric_metric_matrices = []

    for frame in trange(first_frame, last_frame + 1):
        # update locations of geese and plot them
        locations = get_frame_locations(frame, individual_geese_trjs, column_names)

        # create a distance matrix to be filled with dimensions:
        # amount of birds in this frame x amount of birds in this frame
        distance_matrix = np.zeros((locations.shape[0], locations.shape[0]))

        # same with clusteriness matrix
        clusteriness_matrix = np.zeros((locations.shape[0], locations.shape[0]))

        # same with acceleration_clusteriness matrix
        acceleration_clusteriness_matrix = np.zeros(
            (locations.shape[0], locations.shape[0])
        )

        # same with asymetric metric matrix
        asymetric_metric_matrix = np.zeros((locations.shape[0], locations.shape[0]))

        # array of geese indexed by trj_id
        geese_trj_ids = {}
        geese_positions = {}
        geese_velocities = {}
        geese_accelerations = {}

        if type(locations) == list:
            location_plotter = ax.scatter([], [], [], color="red")
        else:
            # determining indexing of birds in dicts
            i = 0
            # iterate through geese and collect them in a dict
            for index, data in locations.iterrows():
                trj_id, x, y, z, xvel, yvel, zvel, xi, eta, zeta = data[column_names]
                # store data in dicts
                geese_trj_ids[i] = trj_id
                geese_positions[i] = np.array([x, y, z])
                geese_velocities[i] = np.array([xvel, yvel, zvel])
                geese_accelerations[i], orthogonal_basis = acceleration_cartesian(
                    geese_velocities[i], xi, eta, zeta
                )

                i += 1

            # iterate through geesee and calculate distances
            for trj_1 in geese_positions:

                trj_id = int(trj_1)

                for trj_2 in geese_positions:

                    other_trj_id = int(trj_2)

                    if trj_id == other_trj_id:
                        pass

                    # if distance_matrix[trj_id][other_trj_id] == 0:
                    else:
                        (
                            clusteriness,
                            acceleration_clusteriness,
                            distance,
                            asymetric_metric,
                        ) = calculate_clusteriness_and_distance(
                            pos_1=geese_positions[trj_id],
                            pos_2=geese_positions[other_trj_id],
                            vel_1=geese_velocities[trj_id],
                            vel_2=geese_velocities[other_trj_id],
                            accel_1=geese_accelerations[trj_id],
                            accel_2=geese_accelerations[other_trj_id],
                        )

                        # track distance in matrix (symmetrical)
                        distance_matrix[trj_id][other_trj_id] = distance

                        # track clusteriness in matrix (symmetrical)
                        clusteriness_matrix[trj_id][other_trj_id] = clusteriness

                        # track acceleration clusteriness in matrix (symmetrical)
                        acceleration_clusteriness_matrix[trj_id][
                            other_trj_id
                        ] = acceleration_clusteriness

                        # track asymetric metric
                        asymetric_metric_matrix[trj_id][other_trj_id] = asymetric_metric

        distance_matrices.append(distance_matrix)
        clusteriness_matrices.append(clusteriness_matrix)
        acceleration_clusteriness_matrices.append(acceleration_clusteriness_matrix)
        asymetric_metric_matrices.append(asymetric_metric_matrix)

    with open("distance_matrices.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i, matrix in enumerate(distance_matrices):
            writer.writerows(matrix)
            writer.writerow([])

    with open("clusteriness_matrices.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i, matrix in enumerate(clusteriness_matrices):
            writer.writerows(matrix)
            writer.writerow([])

    with open("acceleration_clusteriness_matrices.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i, matrix in enumerate(acceleration_clusteriness_matrices):
            writer.writerows(matrix)
            writer.writerow([])

    with open("asymetric_metric_matrices.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i, matrix in enumerate(asymetric_metric_matrices):
            writer.writerows(matrix)
            writer.writerow([])


# ====================================================================================================
# setting values to be used from dataset
column_numbers = [0, 1, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18]
column_names = [
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
filename = "C:/Python Projects/tohoku_university/geese_project/data/20201206-S6F1820E1#3S20.trj"
# =====================================================================================================


# creating dataframes from function
df, individual_geese_trjs, n_trjs = read_trajectory_data(
    filename, column_numbers, column_names
)

# defining movie length
first_frame = int(df["frame"].min())
last_frame = int(df["frame"].max())

# launch distance calculations
calculate_min_distances(individual_geese_trjs, first_frame, last_frame, n_trjs)
