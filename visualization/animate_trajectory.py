import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import trange
import os

from data_engineering.clean_trajectory import clean_data

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
        f"data/trajectory_data/{filename}.trj",
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


def plot_distribution(
    values: list, name: str, filename: str, showing=True, saving=False
):
    """Plot the distribution of some values, using a boxplot with KDE"""

    fig = plt.figure(figsize=(8, 5))
    if isinstance(values, list):
        # Flatten each entry and concatenate
        values = np.concatenate([np.ravel(v) for v in values if v is not None])
    elif isinstance(values, np.ndarray):
        values = values.ravel()
    else:
        # Fallback — try to convert to numpy array and flatten
        values = np.ravel(values)

    all_values = values.ravel()

    # ignoring 0 values
    all_values = all_values[all_values > 0]
    all_values = all_values[np.isfinite(all_values)]
    # threshold = np.percentile(all_values, 80)
    # all_values = all_values[all_values <= threshold]

    counts, bins = np.histogram(all_values, bins="scott")

    sns.barplot(x=bins[:-1], y=counts, color="red")
    sns.lineplot(x=bins[:-1], y=counts, color="blue")

    plt.grid(color="lightgrey")

    # figure prettiness
    plt.title(f"Distribution of {name} values")
    plt.xlabel(f"{name}")

    if showing == True:
        plt.show()

    if saving == True:
        os.makedirs(
            os.path.dirname(f"data/{filename}/figs/{name}_distribution.png"),
            exist_ok=True,
        )
        plt.savefig(f"data/{filename}/figs/{name}_distribution.png")
        plt.close()


def plot_metrics_over_time(metrics: list, filename: str, showing=True, saving=False):

    for metric in metrics:
        if isinstance(metric, list):
            # Flatten each entry and concatenate
            metric = np.concatenate([np.ravel(v) for v in values if v is not None])
        elif isinstance(metric, np.ndarray):
            metric = metric.ravel()
        else:
            # Fallback — try to convert to numpy array and flatten
            metric = np.ravel(values)

        metric = metric.ravel()

    fig, ax = plt.subplots(1, len(metrics), figsize=(10, 4))

    frames = np.array([i for i in range(len(metrics[0]))])

    metric_names = ["boltzmann algebraic connectivity", "entropy"]

    for j in range(len(metrics)):
        metric = metrics[j]
        plot_axis = ax[j]
        sns.lineplot(
            x=frames,
            y=metric,
            ax=plot_axis,
            color="green",
        )

        plot_axis.set_title(f"""{metric_names[j]} over time""")
        plot_axis.set_xlabel(f"""frame""")
        plot_axis.set_ylabel(f"""{metric_names[j]}""")
        plot_axis.grid(color="lightgrey")

    plt.tight_layout()

    if showing == True:
        plt.show()

    if saving == True:
        os.makedirs(
            os.path.dirname(f"data/{filename}/figs/metrics_over_time.png"),
            exist_ok=True,
        )
        plt.savefig(f"data/{filename}/figs/metrics_over_time.png")
        plt.close()


# ==========================================================================================
# this is the actual animation part. The rest is data managing and loading
# ==========================================================================================
def animation(
    filename: str,
    metrics: list,
    entropies: np.ndarray,
    camera_elevation=25,
    camera_rotation=-110,
):
    """make an animation of the birds flying"""

    # ======================================================================================
    # gather data for plotting
    # --------------------------------------------------------------------------------------------
    # setting values to be used from dataset
    column_numbers = [0, 1, 6, 7, 8, 12, 13, 14, 15]
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
    ]
    # -------------------------------------------------------------------------------------------

    # creating dataframes from function
    df, individual_geese_trjs = read_trajectory_data(
        filename, column_numbers, column_names
    )

    # clean data
    df, individual_geese_trjs, n_trjs = clean_data(df, individual_geese_trjs)

    # defining boundaries of plot by finding maximum and minimum of values
    world_maximums = df[["xpos", "ypos", "zpos"]].max()
    world_minimums = df[["xpos", "ypos", "zpos"]].min()

    # defining loop length as the middle 75%
    first_frame = int(df["frame"].min())
    last_frame = int(df["frame"].max())
    length = last_frame - first_frame

    buffer_length = length * 0.125

    first_frame += buffer_length
    last_frame -= buffer_length

    first_frame = int(first_frame)
    last_frame = int(last_frame)

    # ======================================================================================
    # animation settings

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1])

    # main 3D animation plot
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax.view_init(elev=camera_elevation, azim=camera_rotation)

    # set plot boundaries and design
    ax.axes.set_xlim3d(left=world_minimums["xpos"], right=world_maximums["xpos"])
    ax.axes.set_ylim3d(bottom=world_minimums["ypos"], top=world_maximums["ypos"])
    ax.axes.set_zlim3d(bottom=world_minimums["zpos"], top=world_maximums["zpos"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid(True)

    ax.set_title(f"""frame:   """)

    # 2D metric plots on the sicde
    if len(metrics) != 2:
        raise Exception("More than 2 metrics, but 2 expected!")

    metric_names = ["boltzmann algebraic connectivity", "entropy"]

    # plot first metric
    ax_metric1 = fig.add_subplot(gs[0, 1])
    metric1 = metrics[0]
    ax_metric1.set_xlim(0, last_frame - first_frame)
    ax_metric1.set_ylim(0, 2)
    ax_metric1.grid("lightgrey")
    ax_metric1.set_title("boltzmann metric")
    (metric1_plotter,) = ax_metric1.plot([], [], color="green")

    # plot second metric
    ax_metric2 = fig.add_subplot(gs[1, 1])
    metric2 = metrics[1]
    ax_metric2.set_xlim(0, last_frame - first_frame)
    ax_metric2.set_ylim(0, 1)
    ax_metric2.grid("lightgrey")
    ax_metric2.set_title("entropy")
    (metric2_plotter,) = ax_metric2.plot([], [], color="blue")

    plt.tight_layout()

    # =====================================================================================

    # dict with entries of shape trj_id: {x_locs: [], y_locs: [], z_locs[]}
    historical_flight_paths = {}
    historical_flight_path_plotters = {}
    location_plotter = ax.scatter([], [], [], color="red")

    # ===================================================================================
    # real animation part
    # ===================================================================================

    with writer.saving(fig, f"data/{filename}/flight_animation.mp4", 200):
        # plotting new frame
        j = -1
        for frame in trange(first_frame, last_frame + 1):

            j += 1

            metric1_plotter.set_data(
                [i for i in range(j)],
                [metric1[i] for i in range(j)],
            )
            metric2_plotter.set_data(
                [i for i in range(j)],
                [metric2[i] for i in range(j)],
            )

            # current location
            locations = get_frame_locations(frame, individual_geese_trjs)

            # if there is no bird movement in a frame, plot nothing
            if type(locations) == list:
                location_plotter._offsets3d = ([], [], [])
            # else, plot all the new positions as dots
            else:
                location_plotter._offsets3d = (
                    locations["xpos"],
                    locations["ypos"],
                    locations["zpos"],
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
                # plot historical flight paths
                for trj_id in historical_flight_paths:
                    trj = historical_flight_paths[trj_id]
                    if trj_id in historical_flight_path_plotters:
                        historical_flight_path_plotters[trj_id].set_data(
                            trj["x_locs"], trj["y_locs"]
                        )
                        historical_flight_path_plotters[trj_id].set_3d_properties(
                            trj["z_locs"]
                        )

                    else:
                        (new_path,) = ax.plot3D(
                            trj["x_locs"],
                            trj["y_locs"],
                            trj["z_locs"],
                            color="blue",
                            linewidth="0.4",
                        )

                        historical_flight_path_plotters[trj_id] = new_path

            ax.set_title(f"""frame: {frame - first_frame}""")

            # recording state
            writer.grab_frame()
