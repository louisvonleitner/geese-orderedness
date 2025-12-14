import pandas as pd
import numpy as np

df = pd.read_csv("/home/louis/geese_project/predicting/results/boosted_trees_maes_12-12_17-47.csv")


df = df.sort_values("mae", ascending=True)

print(df.describe())

print(df.head(10))