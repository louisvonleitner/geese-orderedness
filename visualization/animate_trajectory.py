import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

from data_engineering.clean_trajectory import clean_data

# movie metadata
# ======================================================================================
plt.rcParams["animation.ffmpeg_path"] = plt.rcParams['animation.ffmpeg_path'] = '/home/louis/bin/ffmpeg'

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


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale so that 1m in x, y, z looks the same."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot radius — half the largest range
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_distribution(metric: dict, filename: str, showing=True, saving=False):
    """Plot the distribution of some values, using a boxplot with KDE"""

    values = metric["values"]
    name = metric["name"]

    fig = plt.figure(figsize=(8, 5))
    if isinstance(values, list):
        # Flatten each entry and concatenate
        values = np.concatenate([np.ravel(v) for v in values if v is not None])
    elif isinstance(values, np.ndarray):
        values = values.ravel()
    else:
        # Fallback — try to convert to numpy array and flatten
        values = np.ravel(values)

    mean = np.mean(values)
    median = np.median(values)
    std_dev = np.std(values)

    fig = plt.figure(figsize=(8, 6), dpi=300)

    ax = sns.histplot(values, color="blue", kde=True, zorder=2)

    for line in ax.lines:
        line.set_color("crimson")

    plt.grid(color="lightgrey", zorder=0)

    ax.axvline(median, color="green", label="median", linestyle="--", zorder=3)
    ax.axvline(mean, color="orange", label="mean", linestyle="-", zorder=3)
    ax.axvspan(
        mean - std_dev,
        mean + std_dev,
        color="orange",
        alpha=0.2,
        zorder=1,
        label="std_dev",
    )

    # figure prettiness
    plt.title(f"Distribution of {name} values")
    plt.xlabel(f"{name}")
    plt.ylabel("count")
    plt.legend()

    plt.tight_layout()

    if showing == True:
        plt.show()

    if saving == True:
        os.makedirs(
            os.path.dirname(f"data/{filename}/figs/{name}_distribution.png"),
            exist_ok=True,
        )
        plt.savefig(f"data/{filename}/figs/{name}_distribution.png")
        plt.close()


def plot_metrics_over_time(
    order_metrics: list, filename: str, showing=True, saving=False
):
    """
    Plots all metrics in a dynamic grid layout that adapts to the number of metrics.

    Parameters
    ----------
    order_metrics : list[dict]
        Each dict must have keys: 'name', 'values', 'color', and optionally 'value_space'
    filename : str
        Base name for saving the plot
    showing : bool
        Whether to display the plot
    saving : bool
        Whether to save the plot to data/{filename}/figs/metrics_over_time.png
    """

    n_metrics = len(order_metrics)
    if n_metrics == 0:
        print("No metrics to plot.")
        return

    for metric in order_metrics:
        metric['values'] = np.nan_to_num(metric['values'], nan=0.0)

    # --- determine grid size dynamically ---
    n_cols = math.ceil(math.sqrt(n_metrics))
    n_rows = math.ceil(n_metrics / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    ax = np.array(ax).reshape(-1)  # flatten in case of single row/col

    frames = np.arange(len(order_metrics[0]["values"]))

    # --- plot each metric ---
    for metric_id, metric in enumerate(order_metrics):
        plot_axis = ax[metric_id]

        sns.lineplot(
            x=frames,
            y=metric["values"],
            ax=plot_axis,
            color=metric.get("color", "C0"),
        )

        # handle value range
        if metric.get("value_space"):
            value_space = metric["value_space"]
            margin = 0.1 * (value_space[1] - value_space[0])
            plot_axis.set_ylim(value_space[0] - margin, value_space[1] + margin)
        else:
            ymin, ymax = np.min(metric["values"]), np.max(metric["values"])
            margin = 0.1 * (ymax - ymin)
            plot_axis.set_ylim(ymin - margin, ymax + margin)

        plot_axis.set_title(f"{metric['name']} over time")
        plot_axis.set_xlabel("Frame")
        plot_axis.set_ylabel(metric["name"])
        plot_axis.grid(color="lightgrey")

    # hide any unused subplots
    for j in range(metric_id + 1, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()

    # --- display or save ---
    if showing:
        plt.show()

    if saving:
        save_path = f"data/{filename}/figs/metrics_over_time.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()


"""
def plot_metrics_over_time(
    order_metrics: list, filename: str, showing=True, saving=False
):

    fig, ax = plt.subplots(2, 2, figsize=(10, 4))

    frames = np.array([i for i in range(len(order_metrics[0]["values"]))])

    for metric_id in range(len(order_metrics)):

        i = metric_id % 2

        if metric_id >= 2:
            j = 1
        else:
            j = 0

        metric = order_metrics[metric_id]

        plot_axis = ax[i][j]
        sns.lineplot(
            x=frames,
            y=metric["values"],
            ax=plot_axis,
            color=metric["color"],
        )
        if metric["value_space"] != []:
            value_space = metric["value_space"]
            plot_axis.set_ylim(value_space[0], value_space[1])
            value_space_length = value_space[1] - value_space[0]
            plot_axis.set_ylim(
                value_space[0] - 0.1 * value_space_length,
                value_space[1] + 0.1 * value_space_length,
            )
        else:
            plot_axis.set_ylim(np.min(metric["values"]), np.max(metric["values"]))

        plot_axis.set_title(f"{metric['name']} over time")
        plot_axis.set_xlabel(f"frame")
        plot_axis.set_ylabel(f"{metric['name']}")
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
"""

# ==========================================================================================
# this is the actual animation part. The rest is data managing and loading
# ==========================================================================================
import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import trange


def animation(
    filename: str,
    order_metrics: list,
    camera_elevation=25,
    camera_rotation=-110,
):
    """Make an animation of the birds flying, with dynamic metric plots on the right."""

    # ======================================================================================
    # Gather data for plotting
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

    df, individual_geese_trjs = read_trajectory_data(
        filename, column_numbers, column_names
    )
    df, individual_geese_trjs, n_trjs = clean_data(df, individual_geese_trjs)

    world_maximums = df[["xpos", "ypos", "zpos"]].max()
    world_minimums = df[["xpos", "ypos", "zpos"]].min()

    first_frame = int(df["frame"].min())
    last_frame = int(df["frame"].max())
    length = last_frame - first_frame
    buffer_length = length * 0.125
    first_frame += buffer_length
    last_frame -= buffer_length
    first_frame, last_frame = int(first_frame), int(last_frame)

    # ======================================================================================
    # Dynamic layout for animation (left) and metrics (right)
    n_metrics = len(order_metrics)
    if n_metrics == 0:
        print("No metrics to plot.")
        return

    # Determine metric grid dynamically
    n_cols = math.ceil(math.sqrt(n_metrics))
    n_rows = math.ceil(n_metrics / n_cols)

    # Make main figure
    fig = plt.figure(figsize=(12, max(6, 3 * n_rows)))
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[2] + [1] * n_cols,
        height_ratios=[1] * n_rows,
    )

    # ======================================================================================
    # LEFT SIDE (MAIN ANIMATION)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.view_init(elev=camera_elevation, azim=camera_rotation)

    ax3d.set_xlim(world_minimums["xpos"], world_maximums["xpos"])
    ax3d.set_ylim(world_minimums["ypos"], world_maximums["ypos"])
    ax3d.set_zlim(world_minimums["zpos"], world_maximums["zpos"])
    set_axes_equal(ax3d)

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("frame: ")
    plt.grid(True)

    # ======================================================================================
    # RIGHT SIDE (Dynamic grid of metrics)
    frames = np.arange(last_frame - first_frame)
    plotters = []
    metric_axes = []

    for i, metric in enumerate(order_metrics):
        # Convert 1D index to grid position
        row = i // n_cols
        col = (i % n_cols) + 1  # +1 because col 0 is 3D plot
        axm = fig.add_subplot(gs[row, col])
        metric_axes.append(axm)

        plotter = sns.lineplot(
            x=[0],
            y=[0],
            ax=axm,
            color=metric.get("color", "C0"),
        )
        plotters.append(plotter.lines[0])

        # Y-limits
        if metric.get("value_space"):
            lo, hi = metric["value_space"]
            margin = 0.1 * (hi - lo)
            axm.set_ylim(lo - margin, hi + margin)
        else:
            vals = np.array(metric["values"])
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            margin = 0.1 * (hi - lo)
            axm.set_ylim(lo - margin, hi + margin)

        axm.set_xlim(0, last_frame - first_frame)
        axm.set_title(metric["name"])
        axm.set_xlabel("frame")
        axm.set_ylabel(metric["name"])
        axm.grid(color="lightgrey")

    # Hide any unused slots in the grid
    total_slots = n_rows * n_cols
    if n_metrics < total_slots:
        for j in range(n_metrics, total_slots):
            fig.add_subplot(gs[j // n_cols, (j % n_cols) + 1]).set_visible(False)

    plt.tight_layout()

    # ======================================================================================
    # Set up data structures for the animation
    historical_flight_paths = {}
    historical_flight_path_plotters = {}
    location_plotter = ax3d.scatter([], [], [], color="red")

    # ======================================================================================
    # Animation writing
    with writer.saving(fig, f"data/{filename}/flight_animation.mp4", 200):
        j = -1
        for frame in range(first_frame, last_frame + 1):
            j += 1

            # Update each metric line
            for k, plotter in enumerate(plotters):
                metric = order_metrics[k]
                plotter.set_data(
                    np.arange(j),
                    metric["values"][:j],
                )

            # Current locations
            locations = get_frame_locations(frame, individual_geese_trjs)
            if isinstance(locations, list):
                location_plotter._offsets3d = ([], [], [])
            else:
                location_plotter._offsets3d = (
                    locations["xpos"],
                    locations["ypos"],
                    locations["zpos"],
                )

                # Update flight paths
                for _, col in locations.iterrows():
                    tid = col["trj_id"]
                    if tid not in historical_flight_paths:
                        historical_flight_paths[tid] = {"x": [], "y": [], "z": []}
                    trj = historical_flight_paths[tid]
                    trj["x"].append(col["xpos"])
                    trj["y"].append(col["ypos"])
                    trj["z"].append(col["zpos"])

                # Draw paths
                for tid, trj in historical_flight_paths.items():
                    if tid in historical_flight_path_plotters:
                        line = historical_flight_path_plotters[tid]
                        line.set_data(trj["x"], trj["y"])
                        line.set_3d_properties(trj["z"])
                    else:
                        (new_line,) = ax3d.plot3D(
                            trj["x"], trj["y"], trj["z"], color="blue", linewidth=0.4
                        )
                        historical_flight_path_plotters[tid] = new_line

            ax3d.set_title(f"frame: {frame - first_frame}")
            writer.grab_frame()


"""
def animation(
    filename: str,
    order_metrics: list,
    camera_elevation=25,
    camera_rotation=-110,
):
    """ """make an animation of the birds flying""" """

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


    n_metrics = len(order_metrics)
    if n_metrics == 0:
        print("No metrics to plot.")
        return

    # --- determine grid size dynamically ---
    n_cols = math.ceil(math.sqrt(n_metrics))
    n_rows = math.ceil(n_metrics / n_cols)

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2, 1, 1], height_ratios=[1, 1])

    # ===========================================================
    # LEFT SIDE (MAIN ANIMATION)

    # main 3D animation plot
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax.view_init(elev=camera_elevation, azim=camera_rotation)

    # set plot boundaries and design
    ax.axes.set_xlim3d(left=world_minimums["xpos"], right=world_maximums["xpos"])
    ax.axes.set_ylim3d(bottom=world_minimums["ypos"], top=world_maximums["ypos"])
    ax.axes.set_zlim3d(bottom=world_minimums["zpos"], top=world_maximums["zpos"])

    # force qual scaling on axes:
    set_axes_equal(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid(True)

    ax.set_title(f"frame:   ")

    # =======================================================
    # 2D order metric plots on the side
    plotters = []
    axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    for metric_id, ax_metric in enumerate(axes):
        if metric_id >= len(order_metrics):
            raise Exception("More than 4 metrics, but 4 expected!")

        metric = order_metrics[metric_id]

        plotter = sns.lineplot(
            x=[0],
            y=[0],
            ax=ax_metric,
            color=metric["color"],
        )
        plotters.append(plotter.lines[0])

        if metric["value_space"] != []:
            value_space = metric["value_space"]
            value_space_length = value_space[1] - value_space[0]
            ax_metric.set_ylim(
                value_space[0] - 0.1 * value_space_length,
                value_space[1] + 0.1 * value_space_length,
            )
        else:
            ax_metric.set_ylim(np.min(metric["values"]), np.max(metric["values"]))

        ax_metric.set_xlim(0, last_frame - first_frame)

        ax_metric.set_title(f"{metric['name']}")
        ax_metric.set_xlabel("frame")
        ax_metric.set_ylabel(metric["name"])
        ax_metric.grid(color="lightgrey")

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

            for plotter_id in range(len(plotters)):
                plotters[plotter_id].set_data(
                    [i for i in range(j)],
                    [order_metrics[plotter_id]["values"][i] for i in range(j)],
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

            ax.set_title(f"frame: {frame - first_frame}")

            # recording state
            writer.grab_frame()

"""
