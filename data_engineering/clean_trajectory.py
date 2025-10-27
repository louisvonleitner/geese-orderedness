import pandas as pd
import numpy as np

"""
def read_trajectory_data(filename, column_numbers, column_names):
    """ """Read data from file into a pandas dataframe and return this dataframe""" """

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
filename = "20201206-S6F1820E1_3S20"
"""


def clean_data(df: pd.DataFrame, individual_geese_trjs: list):

    first_frame, last_frame = np.min(df["frame"]), np.max(df["frame"])
    video_length = last_frame - first_frame

    # if bird is not there for more than 75% of the video
    individual_geese_trjs = [
        trj for trj in individual_geese_trjs if trj.shape[0] >= (video_length * 0.8)
    ]

    if len(individual_geese_trjs) > 0:
        cleaned_df = pd.concat(individual_geese_trjs)

    else:
        return df, individual_geese_trjs, 0

    n_trjs = len(individual_geese_trjs)

    return cleaned_df, individual_geese_trjs, n_trjs
