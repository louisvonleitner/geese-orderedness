import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


# 2. Define the Combined Schedule Function
def get_cosine_with_warmup_scheduler(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    base_lr,
    min_lr=0,
):

    def lr_lambda(current_step):

        # A. Warmup Phase (Linear increase from 0 to 1.0)
        if current_step < num_warmup_steps:
            # Scale factor grows linearly: 0 -> 1.0
            return float(current_step) / float(max(1, num_warmup_steps))

        # B. Cosine Annealing Phase (Decay from 1.0 to min_factor)
        else:
            # Number of steps for the decay phase
            num_decay_steps = num_training_steps - num_warmup_steps

            # Step progress within the decay phase (0 to 1.0)
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_decay_steps)
            )
            progress_tensor = torch.tensor(progress)

            # The standard Cosine Annealing formula
            cosine_output = 0.5 * (1.0 + torch.cos(torch.pi * progress_tensor))

            # Calculate the minimum factor (Min_LR / BASE_LR)
            min_factor = min_lr / base_lr if base_lr > 0 else 0

            # Interpolate between 1.0 and min_factor
            return min_factor + (1.0 - min_factor) * cosine_output

    return LambdaLR(optimizer, lr_lambda)
