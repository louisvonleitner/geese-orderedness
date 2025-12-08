import pandas as pd
import numpy as np


# ==============================================================================


def read_trajectory_data(filepath: str, column_numbers: list, column_names: list):
    """
    Read data from file into a pandas dataframe and return.

    Read data from file.
    Parse seperate trajectories.

    Parameters
    ----------
    filepath: str
        Path to file to be read.
    column_numbers:
        Numbers of relevant Columns to be read. Defined by .trj file structure.
    column_names:
        Names according to columns of .trj structure files.

    Return
    ------
    DataFrame
        All positions of all geese at all times.
    array_like
        List of DataFrames of individual geese trajectories.
    int
        Number of different geese in file.
    """

    # dataframe including all trajectories
    df = pd.read_csv(
        filepath,
        sep="\s+",
        usecols=column_numbers,
        names=column_names,
        dtype=np.float32,
    )

    df["trj_id"] = df["trj_id"].astype(int)

    # get number of trajectories
    n_trjs = len(df["trj_id"].unique())

    return df, n_trjs


def clean_data(df: pd.DataFrame):
    """
    Clean a trajectory DataFrame, filtering out trajectories and adjusting length.

    Remove the first and last 50 frames of the video.
    Remove geese trajectories that are shorter than 80% of the video.

    Parameters
    ----------
    df : DataFrame
        Raw dataframe read from .trj file.

    Returns
    -------
    DataFrame
        The cleaned DataFrame including all valid geese trajectories.
    array_like
        List of all valid individual geese trajectories.
    int
        Number of valid trajectories == Number of geese.
    int
        Length of filtered video in number of frames.

    """

    first_frame, last_frame = np.min(df["frame"]), np.max(df["frame"])
    initial_video_length = last_frame - first_frame

    # cut off beginning and end of video for more clean data
    start_frame = first_frame + 50
    end_frame = last_frame - 50
    new_video_length = end_frame - start_frame

    # do not use videos shorter than 100 frames
    # if new_video_length < 100:
    # raise Exception("Video too short")

    # seperate trajectories extracted
    individual_geese_trjs = [group_df for trj_id, group_df in df.groupby("trj_id")]
    # if bird is not there for more than 80% of the video
    individual_geese_trjs = [
        trj for trj in individual_geese_trjs if trj.shape[0] >= (new_video_length * 0.8)
    ]

    if len(individual_geese_trjs) > 0:
        cleaned_df = pd.concat(individual_geese_trjs)

    else:
        return df, individual_geese_trjs, 0

    n_trjs = len(individual_geese_trjs)

    return cleaned_df, n_trjs, new_video_length


def load_and_clean_trajectory(trj_path: str):
    """
    Load and clean trajectory.

    Load trajectory from string.
    Clean trajectory.

    Parameters
    ----------
    trj_path : str
        Full path to trajectory file.

    Return
    ------
    DataFrame
        Dataframe including all positions of all geese at all frames.
    array_like
        List of all sepeprate geese trajectories.
    int
        Number of different geese in video.
    int
        Length of video in number of frames.
    """

    # ==============================================================
    # define column numbers and name based on .trj format
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

    # =============================================================

    # extract data from file
    df, n_trjs_before_cleaning = read_trajectory_data(
        filepath=trj_path,
        column_numbers=column_numbers,
        column_names=column_names,
    )

    df, n_trjs, video_length = clean_data(
        df=df,
    )

    print(
        f"Cleaning removed {n_trjs_before_cleaning - n_trjs} / {n_trjs_before_cleaning} bad trajectories."
    )

    # add data to dataframe
    df["n_geese_whole_trj"] = n_trjs
    df["video_length"] = video_length

    return df
