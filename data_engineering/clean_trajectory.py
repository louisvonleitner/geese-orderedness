import pandas as pd
import numpy as np


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
