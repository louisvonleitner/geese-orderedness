import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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


# ===================================================================================================
# TSNE PART
# ===================================================================================================


def apply_tsne(frame, individual_geese_trjs):
    column_names = [
        "trj_id",
        "frame",
        "xpos",
        "ypos",
        "zpos",
    ]

    # get locations
    locations = get_frame_locations(frame, individual_geese_trjs, column_names)

    positions = locations[["xpos", "ypos", "zpos"]].to_numpy()

    tsne = TSNE(
        n_components=2,
        perplexity=5,
        random_state=42,
        max_iter=10000,
        learning_rate="auto",
    )

    embedded_positions = tsne.fit_transform(positions)

    # --- Plot before (3D) and after (2D) ---
    fig = plt.figure(figsize=(12, 5))

    # Original 3D
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=80, c="steelblue")
    ax1.set_title("Original 3D Data")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("X3")

    # t-SNE 2D
    ax2 = fig.add_subplot(122)
    ax2.scatter(embedded_positions[:, 0], embedded_positions[:, 1], s=80, c="crimson")
    ax2.set_title("t-SNE Embedding (3D → 2D)")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.show()


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


# launch distance calculations
apply_tsne(2100, individual_geese_trjs)
