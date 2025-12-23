import os
import shutil
import random

# 1. Setup paths
source_dir = "../data/trajectory_data"
train_dir = "../data/train_data/number_of_geese_training/train_data"
val_dir = "../data/train_data/number_of_geese_training/val_data"
test_dir = "../data/train_data/number_of_geese_training/test_data"

# Create folders if they don't exist
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# 2. Get and shuffle files
files = [
    f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
]
random.seed(42)  # For reproducibility
random.shuffle(files)

# 3. Calculate split indices
total = len(files)
train_end = int(total * 0.7)
val_end = int(total * (0.7 + 0.15))  # 15% validation

# 4. Distribute files
for i, file_name in enumerate(files):
    src_path = os.path.join(source_dir, file_name)

    if i < train_end:
        dest_path = os.path.join(train_dir, file_name)
    elif i < val_end:
        dest_path = os.path.join(val_dir, file_name)
    else:
        dest_path = os.path.join(test_dir, file_name)

    shutil.copy(
        src_path, dest_path
    )  # Use shutil.move() if you want to delete originals

print(f"Done! Processed {total} files.")
