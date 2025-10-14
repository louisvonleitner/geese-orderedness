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


def calculate_clusteriness_and_distance(pos_1, pos_2, vel_1, vel_2):
    """calculate and return the clusteriness and distance between two birds"""
    distance = np.linalg.norm(pos_1 - pos_2)
    dot_prod = np.dot(vel_1, vel_2)

    norming_factor = np.linalg.norm(vel_1) * np.linalg.norm(vel_2)
    if norming_factor != 0:
        clusteriness = (1 / distance) * (
            dot_prod / (np.linalg.norm(vel_1) * np.linalg.norm(vel_2))
        )

    # division by 0 avoided (unclear speed = 0 !)
    else:
        clusteriness = 0

    return clusteriness, distance


def calculate_min_distances(individual_geese_trjs, first_frame, last_frame, n_trjs):

    column_names = ["trj_id", "xpos", "ypos", "zpos", "xvel", "yvel", "zvel"]

    distance_matrices = []
    clusteriness_matrices = []

    for frame in trange(first_frame, last_frame + 1):
        # update locations of geese and plot them
        locations = get_frame_locations(frame, individual_geese_trjs, column_names)

        # create a distance matrix to be filled with dimensions:
        # amount of birds x amount of birds
        distance_matrix = np.zeros((n_trjs, n_trjs))

        # same with clusteriness matrix
        clusteriness_matrix = np.zeros((n_trjs, n_trjs))

        # array of geese indexed by trj_id
        geese_positions = {}
        geese_velocities = {}

        if type(locations) == list:
            location_plotter = ax.scatter([], [], [], color="red")
        else:
            # iterate through geese and collect them in a dict
            for index, data in locations.iterrows():
                trj_id, x, y, z, xvel, yvel, zvel = data[
                    ["trj_id", "xpos", "ypos", "zpos", "xvel", "yvel", "zvel"]
                ]
                # store data in dicts
                geese_positions[trj_id] = np.array([x, y, z])
                geese_velocities[trj_id] = np.array([xvel, yvel, zvel])

            # iterate through geesee and calculate distances
            for trj_1 in geese_positions:

                trj_id = int(trj_1)

                for trj_2 in geese_positions:

                    other_trj_id = int(trj_2)

                    if trj_id == other_trj_id:
                        pass

                    elif distance_matrix[trj_id][other_trj_id] != 0:
                        pass

                    # if distance_matrix[trj_id][other_trj_id] == 0:
                    else:
                        clusteriness, distance = calculate_clusteriness_and_distance(
                            pos_1=geese_positions[trj_id],
                            pos_2=geese_positions[other_trj_id],
                            vel_1=geese_velocities[trj_id],
                            vel_2=geese_velocities[other_trj_id],
                        )

                        # track distance in matrix (symmetrical)
                        distance_matrix[trj_id][other_trj_id] = distance
                        distance_matrix[other_trj_id][trj_id] = distance

                        # track clusteriness in matrix (symmetrical)
                        clusteriness_matrix[trj_id][other_trj_id] = clusteriness
                        clusteriness_matrix[other_trj_id][trj_id] = clusteriness

        distance_matrices.append(distance_matrix)
        clusteriness_matrices.append(clusteriness_matrix)

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


# ====================================================================================================
# setting values to be used from dataset
column_numbers = [0, 1, 6, 7, 8, 12, 13, 14, 15]
column_names = ["trj_id", "frame", "xpos", "ypos", "zpos", "xvel", "yvel", "zvel", "n"]
filename = "C:/Python Projects/tohoku_university/geese_project/data/20201206-S6F1820E1#3S20.trj"
# =====================================================================================================


# creating dataframes from function
df, individual_geese_trjs, n_trjs = read_trajectory_data(
    filename, column_numbers, column_names
)

# defining boundaries of plot by finding maximum and minimum of values
world_maximums = df[["xpos", "ypos", "zpos"]].max()
world_minimums = df[["xpos", "ypos", "zpos"]].min()

# defining movie length
first_frame = int(df["frame"].min())
last_frame = int(df["frame"].max())

# launch distance calculations
calculate_min_distances(individual_geese_trjs, first_frame, last_frame, n_trjs)
