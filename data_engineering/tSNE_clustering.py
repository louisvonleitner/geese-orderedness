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
    return a_cartesian


def convert_acceleration(df):

    for index, row in df.iterrows():
        xi, eta, zeta = row[["xi", "eta", "zeta"]]
        v = np.array([row["xvel"], row["yvel"], row["zvel"]])
        cartesian_acceleration = acceleration_cartesian(v, xi, eta, zeta)
        xacc, yacc, zacc = cartesian_acceleration
        df.loc[index, ["xacc", "yacc", "zacc"]] = xacc, yacc, zacc

    return df


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
        "xvel",
        "yvel",
        "zvel",
        "xi",
        "eta",
        "zeta",
    ]

    # get locations
    locations = get_frame_locations(frame, individual_geese_trjs, column_names)

    locations = convert_acceleration(locations)

    X = locations[
        ["xpos", "ypos", "zpos", "xvel", "yvel", "zvel", "xacc", "yacc", "zacc"]
    ].to_numpy()

    # print("Starting PCA")
    # pca = PCA(n_components=0.95, random_state=42)
    # X_pca = pca.fit_transform(X)
    # print(f"PCA reduced shape: {X_pca.shape}")

    tsne = TSNE(
        n_components=2,
        perplexity=4,
        random_state=42,
        max_iter=3000,
        learning_rate="auto",
        early_exaggeration=12.0,
    )

    print("starting tSNE")
    embedded_X = tsne.fit_transform(X)

    # coloring
    mask = X[:, 2] < 34

    # --- Plot before (3D) and after (2D) ---
    fig = plt.figure(figsize=(12, 5))

    # Original 3D
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(X[~mask, 0], X[~mask, 1], X[~mask, 2], s=80, c="green")
    ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], s=80, c="red")
    ax1.set_title("Original 3D Data")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("X3")

    # t-SNE 2D
    ax2 = fig.add_subplot(122)
    ax2.scatter(embedded_X[~mask, 0], embedded_X[~mask, 1], s=80, c="green")
    ax2.scatter(embedded_X[mask, 0], embedded_X[mask, 1], s=80, c="red")
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
apply_tsne(2150, individual_geese_trjs)
