from predicting.boosted_tree_training import tree_training_wrapper, grid_search
import torch

filepath = "data/training_data/number_of_geese.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

grid_search(filepath)
# tree_training_wrapper(filepath)
