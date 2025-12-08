import numpy as np
import pandas as pd

from data_engineering.cleaning import load_and_clean_trajectory

from data_engineering.metric_calculation import (
    calculate_velocity_alignment,
    calculate_velocity_deviation,
    calculate_velocity_PCA,
    gaussian_entropy,
    flight_deviations,
)


def acceleration_cartesian(v: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
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

    xi, eta, zeta = acceleration

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


def make_frame_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn trajectory DataFrame into a DataFrame with one row for each frame.

    Turn into one row per frame.

    Parameters
    ----------
    df : DataFrame
        Trajectory based DataFrame.

    Return
    ------
    DataFrame
        DataFrame containing one row per frame with basic information about geese.
    """

    df["position"] = list(df[["xpos", "ypos", "zpos"]].values)
    df["velocity"] = list(df[["xvel", "yvel", "zvel"]].values)
    df["acceleration"] = list(df[["xi", "eta", "zeta"]].values)

    # create DataFrame with a row for each frame
    grouped_frames = df.groupby("frame")

    frame_data = grouped_frames.agg(
        # Count the number of geese in the frame
        n_geese=("trj_id", "count"),
        # calculate mean velocity
        # Aggregate all geese positional data
        geese_positions=("position", list),
        geese_velocities=("velocity", list),
        geese_accelerations=("acceleration", list),
    )

    # compute mean velocity vector
    # Apply np.mean to the list of vectors in each cell
    # axis=0 collapses the N geese into 1 average vector
    frame_data["average_velocity"] = frame_data["geese_velocities"].apply(
        lambda geese_list: (
            np.mean(np.stack(geese_list), axis=0)
            if len(geese_list) > 0
            else np.zeros(3)
        )
    )

    # transform relative acceleration into cartesian coordinate system
    # We use zip() to pair the specific velocity and acceleration list for each row simultaneously.
    frame_data["geese_accelerations"] = [
        [
            acceleration_cartesian(row_avg_vel, single_goose_acc)
            for single_goose_acc in row_acc_list
        ]
        for row_avg_vel, row_acc_list in zip(
            frame_data["average_velocity"], frame_data["geese_accelerations"]
        )
    ]

    return frame_data


def apply_metrics(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply metrics to frame and add them as feature columns.

    Parameters
    ----------
    frame_data : DataFrame
        Frame based DataFrame where each row represents a timeframe
        containing lists of geese positions/velocities.

    Return
    ------
    DataFrame
        DataFrame with added column features.
    """

    # 1. Define the metric functions dictionary
    metrics = {
        "velocity_alignment": calculate_velocity_alignment,
        "velocity_deviation": calculate_velocity_deviation,
        "pca": calculate_velocity_PCA,
        "gaussian_entropy": gaussian_entropy,
        "flight_deviations": flight_deviations,
    }

    # 2. Define a helper function to process a SINGLE row
    def process_row(row):
        geese = {}

        # safely get the number of geese for this specific row
        n_geese = row["n_geese"]

        # Build the geese dictionary for this frame
        for i in range(n_geese):
            position = row["geese_positions"][i]
            velocity = row["geese_velocities"][i]

            goose = {
                "trj_id": i,
                "position": position,
                "velocity": velocity,
                "velocity_norm": np.linalg.norm(velocity),
            }
            geese[i] = goose

        # Calculate all metrics for this frame
        row_results = {}
        for name, func in metrics.items():
            # Apply the function to the geese dict we just built
            row_results[name] = func(geese=geese)

        return pd.Series(row_results)

    # 3. Apply the helper to every row (axis=1)
    # This returns a new DataFrame containing only the metric columns
    metric_features = frame_data.apply(process_row, axis=1)

    # 4. Join the new metrics back to the original DataFrame
    result_df = pd.concat([frame_data, metric_features], axis=1)

    return result_df


def engineer_trajectory_data(filepath: str):
    """
    Engineer trajectory file to data that is suitalbe for model training.

    Load and clean data.
    Group data by frame instead of geese.

    Parameters
    ----------
    filepath : str
        Path to the trajectory file.

    Return
    ------
    Write data to new .csv file for later use.
    """
    # load and clean trajectories. now sorted by geese
    df = load_and_clean_trajectory(filepath)

    # turn into frame based DataFrame
    frame_data = make_frame_based(df=df)

    frame_data = apply_metrics(frame_data)

    return frame_data
