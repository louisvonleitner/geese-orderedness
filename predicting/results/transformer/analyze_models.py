import pandas as pd
import numpy as np

df = pd.read_csv(
    "/home/louis/geese_project/predicting/results/transformer/model_params.csv"
)

print(df)
print(df.head(5))

sorted_df = df.sort_values("val_loss", ascending=True)
print(sorted_df.head(5))
for i in range(5):
    print("==========================================================")
    print(i)
    print("----------------------------------------------------------")
    print(sorted_df.iloc[i])
