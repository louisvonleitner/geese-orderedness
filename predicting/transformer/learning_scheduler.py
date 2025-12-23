import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

import numpy as np


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


class EarlyStopper:
    """
    Implements early stopping and tracks the generalization gap
    (Val Loss - Train Loss) as a metric for overfitting.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience  # The number of epochs (k) to wait
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf  # Best validation loss observed
        self.stopped_epoch = 0
        self.is_stopped = False

        # New attributes for tracking overfitting
        self.generalization_gap = []
        self.best_gap = np.inf

    def __call__(self, train_loss: float, val_loss: float, epoch: int) -> bool:
        """
        Checks the validation loss and tracks the overfitting generalization gap.

        Args:
            train_loss (float): The current training loss for the epoch.
            val_loss (float): The current validation loss for the epoch.
            epoch (int): The current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.is_stopped:
            return True

        # --- 1. Track Generalization Gap (Overfitting Indicator) ---
        # A large and increasing gap is a strong sign of overfitting.
        current_gap = val_loss - train_loss
        self.generalization_gap.append(current_gap)

        # Optional: You can save the model that had the minimum gap,
        # though minimizing val_loss is usually sufficient.
        self.best_gap = min(self.best_gap, current_gap)

        # --- 2. Early Stopping Logic (Based on Val Loss) ---

        # Improvement found?
        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            self.counter = 0  # Reset patience counter

        # No significant improvement?
        else:
            self.counter += 1

            # Stopping condition met?
            if self.counter >= self.patience:
                self.is_stopped = True
                self.stopped_epoch = epoch
                print(f"\n--- 🛑 Early Stopping triggered at epoch {epoch}! ---")
                print(
                    f"Validation loss has not improved for {self.patience} consecutive epochs."
                )
                return True

        return False

    def get_overfitting_status(self) -> str:
        """Provides a status update based on the tracked generalization gap."""
        if not self.generalization_gap:
            return "No data recorded yet."

        last_gap = self.generalization_gap[-1]

        # Check trend over the last few epochs (e.g., last 3)
        if len(self.generalization_gap) >= 3:
            recent_gaps = np.array(self.generalization_gap[-3:])
            # Check if the gap is consistently increasing
            if (recent_gaps[1] > recent_gaps[0]) and (recent_gaps[2] > recent_gaps[1]):
                trend = "Gap is increasing rapidly (STRONG Overfitting sign)."
            else:
                trend = "Gap is stable or fluctuating."
        else:
            trend = "Not enough data to determine a trend."

        return (
            f"Current Generalization Gap (Val - Train) = **{last_gap:.4f}**\n"
            f"Trend: {trend}"
        )
