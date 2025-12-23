import pandas as pd
import numpy as np

df = pd.read_csv(
    "/home/louis/geese_project/predicting/results/boosted_trees/maes_12-23_12-55.csv"
)


df = df.sort_values("val_r2", ascending=False)
# df = df.sort_values("val_mae", ascending=True)

print(df.describe())

print(df.head(10))
