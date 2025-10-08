import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.animation import FFMpegWriter
from tqdm import trange

from geese import Goose, Flock


# movie metadata
# ======================================================================================
plt.rcParams["animation.ffmpeg_path"] = (
    "C:/Users/luigi/Downloads/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
)
metadata = dict(title="Movie", artist="LouisvonLeitner")
writer = FFMpegWriter(fps=60, metadata=metadata)
# =======================================================================================


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

    # list of dataframes, where one dataframe holds exactly one trajectory
    individual_geese_trjs = [group_df for trj_id, group_df in df.groupby("trj_id")]
    return df, individual_geese_trjs


def get_frame_locations(frame, individual_geese_trjs):
    """return locations of all geese currently visible by the camera for a specific frame
    returns location data as a pandas Dataframe  xloc, yloc, zloc"""

    # column names of the data that should be returned
    return_data_names = ["trj_id", "xpos", "ypos", "zpos"]
    locations = []
    for trj in individual_geese_trjs:
        # grab location data based on frame
        location_data = trj[trj["frame"] == frame]

        if not location_data.empty:
            locations.append(location_data[return_data_names])

    if locations != []:
        locations = pd.concat(locations)

    return locations


# experiment for animation
def animation(
    individual_geese_trjs, first_frame, last_frame, world_maximums, world_minimums
):
    """make an animation of the birds flying"""

    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=40, azim=-60)

    # set plot boundaries and design
    ax.axes.set_xlim3d(left=world_minimums["xpos"], right=world_maximums["xpos"])
    ax.axes.set_ylim3d(bottom=world_minimums["ypos"], top=world_maximums["ypos"])
    ax.axes.set_zlim3d(bottom=world_minimums["zpos"], top=world_maximums["zpos"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid(True)

    # real animation part
    with writer.saving(fig, "flying_geese.mp4", 400):
        for frame in trange(first_frame, last_frame + 1):

            # plotting new frame
            # ------------------------------------------------------------

            # update locations of geese and plot them
            locations = get_frame_locations(frame, individual_geese_trjs)

            if type(locations) == list:
                location_plotter = ax.scatter([], [], [], color="red")
            else:
                # remove goose point if it is not moving anymore
                geese_to_remove = []
                for trj_id in Goose.active_geese:
                    goose = Goose.active_geese[trj_id]
                    if trj_id not in locations["trj_id"]:
                        goose.absent_counter += 1
                        if goose.absent_counter >= 60:
                            geese_to_remove.append(goose)

                for goose in geese_to_remove:
                    goose.deactivate()

                for index, goose_data in locations.iterrows():

                    trj_id = goose_data["trj_id"]

                    # if goose has not been seen before, create a new goose object
                    if trj_id not in Goose.all_geese:
                        Goose(
                            trj_id,
                            goose_data["xpos"],
                            goose_data["ypos"],
                            goose_data["zpos"],
                            ax,
                        )
                    else:
                        Goose.all_geese[trj_id].update_position(
                            goose_data["xpos"], goose_data["ypos"], goose_data["zpos"]
                        )

                    # plot the location
                    goose = Goose.all_geese[trj_id]
                    Goose.all_geese[trj_id].location_plotter.set_data(
                        [goose.position[0]],  # x coordinate
                        [goose.position[1]],  # y coordinate
                    )
                    Goose.all_geese[trj_id].location_plotter.set_3d_properties(
                        [goose.position[2]]  # z coordinate
                    )

                    trj = Goose.all_geese[trj_id].historical_flight_path

                    Goose.all_geese[trj_id].history_plotter.set_data(
                        trj["xlocs"],  # x coordinate
                        trj["ylocs"],  # y coordinate
                    )
                    Goose.all_geese[trj_id].history_plotter.set_3d_properties(
                        trj["zlocs"]  # z coordinate
                    )

                """
                # plot historical flight path
                if len(Goose.all_geese) > 0:
                    for trj_id in Goose.all_geese:
                        trj = Goose.all_geese[trj_id].historical_flight_path

                        Goose.all_geese[trj_id].history_plotter.set_data(
                            trj["xlocs"],  # x coordinate
                            trj["ylocs"],  # y coordinate
                        )
                        Goose.all_geese[trj_id].history_plotter.set_3d_properties(
                            trj["zlocs"]  # z coordinate
                """
            # recording state
            writer.grab_frame()


"""
def animation(
    individual_geese_trjs, first_frame, last_frame, world_maximums, world_minimums
):
    """ """make an animation of the birds flying""" """

    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=30, azim=-60)

    # set plot boundaries and design
    ax.axes.set_xlim3d(left=world_minimums["xpos"], right=world_maximums["xpos"])
    ax.axes.set_ylim3d(bottom=world_minimums["ypos"], top=world_maximums["ypos"])
    ax.axes.set_zlim3d(bottom=world_minimums["zpos"], top=world_maximums["zpos"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid(True)

    # dict with entries of shape trj_id: {x_locs: [], y_locs: [], z_locs[]}
    historical_flight_paths = {}
    # real animation part
    with writer.saving(fig, "flying_geese.mp4", 400):
        for frame in trange(first_frame, last_frame + 1):

            # plotting new frame

            # current location
            locations = get_frame_locations(frame, individual_geese_trjs)

            if type(locations) == list:
                location_plotter = ax.scatter([], [], [], color="red")
            else:
                location_plotter = ax.scatter(
                    locations["xpos"], locations["ypos"], locations["zpos"], color="red"
                )

                # update historical fight paths
                for index, column in locations.iterrows():
                    trj_id = column["trj_id"]
                    # if trajectory id has not been seen yet, add a new dict entry
                    if trj_id not in historical_flight_paths:
                        historical_flight_paths[trj_id] = {
                            "x_locs": [],
                            "y_locs": [],
                            "z_locs": [],
                        }

                    # adding position to historical data
                    historical_flight_paths[trj_id]["x_locs"].append(column["xpos"])
                    historical_flight_paths[trj_id]["y_locs"].append(column["ypos"])
                    historical_flight_paths[trj_id]["z_locs"].append(column["zpos"])

            if len(historical_flight_paths) > 0:
                # plot historical flight path
                drawn_paths = []
                for trj_id in historical_flight_paths:
                    trj = historical_flight_paths[trj_id]
                    new_path = ax.plot3D(
                        trj["x_locs"],
                        trj["y_locs"],
                        trj["z_locs"],
                        color="blue",
                        linewidth="0.4",
                    )

                    drawn_paths.append(new_path)

            # recording state
            writer.grab_frame()

            # removing old frame
            location_plotter.remove()
            for path in drawn_paths:
                path[0].remove()
"""

# ====================================================================================================
# setting values to be used from dataset
column_numbers = [0, 1, 6, 7, 8, 12, 13, 14, 15]
column_names = ["trj_id", "frame", "xpos", "ypos", "zpos", "xvel", "yvel", "zvel", "n"]
filename = "20201206-S6F1820E1#3S20.trj"
# =====================================================================================================


# creating dataframes from function
df, individual_geese_trjs = read_trajectory_data(filename, column_numbers, column_names)

# defining boundaries of plot by finding maximum and minimum of values
world_maximums = df[["xpos", "ypos", "zpos"]].max()
world_minimums = df[["xpos", "ypos", "zpos"]].min()

# defining movie length
first_frame = int(df["frame"].min())
last_frame = int(df["frame"].max())


# launch animation creation

animation(
    individual_geese_trjs=individual_geese_trjs,
    first_frame=first_frame,
    last_frame=last_frame,
    world_maximums=world_maximums,
    world_minimums=world_minimums,
)
