import pandas as pd
import numpy as np

cols = [
    "n_params",
    "d_model",
    "n_known_geese",
    "input_length",
    "n_heads",
    "n_layers",
    "dim_feedforward",
    "activation_fn",
    "bias",
    "lr/max_lr",
    "min_lr",
    "batch_size",
    "epochs",
    "total_steps",
    "warmup_steps",
    "beta_1",
    "beta_2",
    "epsilon",
    "weight_decay",
    "train_loss",
    "val_loss",
    "timestamp",
]
# df = pd.read_csv(
#     "/home/louis/geese_project/predicting/results/transformer/grid_search_model_params.csv",
#     names=cols,
# )
# df = pd.read_csv(
#     "/home/louis/geese_project/predicting/results/transformer/model_params.csv",
# )
df = pd.read_csv(
    "~/geese_project/predicting/results/transformer/small_optuna_optimization_model_params.csv",
    names=cols,
    index_col=False,
)

print(df)
print(df.head(5))

n_cols = len(df.columns)

sorted_df = df.sort_values("val_loss", ascending=True)
print(sorted_df.head(5))
for i in range(5):
    print("==========================================================")
    print(i)
    print("----------------------------------------------------------")
    print(sorted_df.iloc[i])
