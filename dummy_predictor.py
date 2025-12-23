import pandas as pd
import numpy as np

from predicting.train_tools import normalize_dataframe, get_std_n_geese

from sklearn.metrics import mean_absolute_error, max_error, r2_score

target_feature = ["n_geese"]
df = pd.read_pickle(
    "/home/louis/geese_project/data/training_data/number_of_geese_data/val_data.pkl"
)[target_feature]

df_normed = normalize_dataframe(df, target_feature, [])

values = np.array(df_normed["n_geese"].values)

median = np.median(values)
medians = np.array([median for _ in range(len(values))])

mae = mean_absolute_error(values, medians)
max_error = max_error(values, medians)
r2 = r2_score(values, medians)

n_geese_std = get_std_n_geese()

print("MAE: ", mae * n_geese_std)
print("Max Error: ", max_error * n_geese_std)
print("R2: ", r2)
