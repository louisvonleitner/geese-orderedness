import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os
import optuna

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.nn import MSELoss, L1Loss

from predicting.transformer.model import goose_number_transformer
from predicting.transformer.learning_scheduler import (
    get_cosine_with_warmup_scheduler,
    EarlyStopper,
)
from predicting.transformer.data_engineering import create_dataloaders


def train_step(model, optimizer, scheduler, loss_fn, train_loader, device):
    total_loss = 0
    model.train()
    for batch in train_loader:

        # zero gradients
        optimizer.zero_grad()

        # extract batch
        X, y_true = [t.to(device) for t in batch]

        # create token_id mask
        token_type_ids = (
            torch.tensor([0] + [1] * (X.size(1) - 1)).repeat(X.shape[0], 1).to(device)
        )

        # make prediction
        y_pred = model(X, token_type_ids)

        # compute loss
        loss = loss_fn(y_true, y_pred)

        # backward pass
        loss.backward()

        # Gradient Clipping <--- ADD THIS
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # improve weights
        optimizer.step()

        # learning rate scheduler
        scheduler.step()

        total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    return avg_loss


def validate(model, val_loader, loss_fn, device):

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            X, y_true = [t.to(device) for t in batch]

            # create token_id mask
            token_type_ids = (
                torch.tensor([0] + [1] * (X.size(1) - 1))
                .repeat(X.shape[0], 1)
                .to(device)
            )
            # Forward Pass
            y_pred = model(X, token_type_ids)

            # loss computation
            loss = loss_fn(y_true, y_pred)

            # sum losses
            total_val_loss += loss.item() * X.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)

    return avg_val_loss


def save_training(model, train_losses, val_losses, params):

    os.makedirs("./predicting/results/transformer/", exist_ok=True)
    os.makedirs("./predicting/models/transformers/", exist_ok=True)
    time_now = datetime.now().strftime("%m%d_%H%M%S")

    loss_dict = {"train_loss": train_losses, "val_loss": val_losses}
    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(f"./predicting/results/transformer/{time_now}.csv")
    print("Metrics saved", flush=True)

    params["train_loss"] = train_losses[-1]
    params["val_loss"] = val_losses[-1]
    params["timestamp"] = time_now
    model_parameters = pd.DataFrame([params])
    model_parameters.to_csv(
        f"./predicting/results/transformer/5_geese_transformer_model_params.csv",
        mode="a",
        header=False,
        index=False,
    )
    print("Saved parameters", flush=True)

    torch.save(model.state_dict(), f"./predicting/models/transformers/model_{time_now}")
    print("Model saved", flush=True)


def training(params, trial=None):

    num_epochs = 100
    # ------------------------
    # transformer architecture
    n_known_geese = 5
    num_created_samples = 1
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_layers = params["n_layers"]
    # ------------------------
    # defining hyperparameters
    # learning
    batch_size = params["batch_size"]
    lr = params["lr"]
    min_lr = params["min_lr"]
    weight_decay = params["weight_decay"]
    beta_1 = params["beta_1"]
    beta_2 = params["beta_2"]
    dropout = params["dropout"]
    epsilon = 0.000001
    # ------------------------
    # load dataloaders
    print("Loading data")
    train_loader, val_loader = create_dataloaders(
        batch_size, n_known_geese, num_created_samples
    )
    # ------------------------
    # selecting device cuda
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------
    print("Initializing model...", flush=True)
    # initiating model, optimizer, scheduler and loss function
    model = goose_number_transformer(
        n_known_geese=n_known_geese,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    model.to(device)
    # optimizer
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        betas=(beta_1, beta_2),
        eps=epsilon,
        weight_decay=weight_decay,
    )
    # scheduler
    # hyperparameters
    total_steps = int(num_epochs * len(train_loader.dataset) / batch_size)
    warmup_steps = 10000 if total_steps > 10000 else total_steps * 0.1
    scheduler = get_cosine_with_warmup_scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=min_lr,
        base_lr=lr,
    )
    loss_fn = L1Loss()
    # loss_fn = MSELoss()
    # early_stopper = EarlyStopper(patience=early_stopping_patience)
    # ------------------------
    # saving data
    params = {
        "n_params": d_model * 28 + 12 * d_model**2,
        "d_model": d_model,
        "n_known_geese": n_known_geese,
        "input_length": model.input_length,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "dim_feedforward": model.dim_feedforward,
        "activation_fn": model.activation,
        "bias": model.bias,
        "lr/max_lr": lr,
        "min_lr": min_lr,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "epsilon": epsilon,
        "weight_decay": weight_decay,
    }
    # -----------------------
    # logging
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print(
        f"Doing {warmup_steps} warmup steps and {total_steps} total training steps over {num_epochs} epochs",
        flush=True,
    )
    # ---------------------
    # train loop
    for epoch in range(num_epochs):

        train_loss = train_step(
            model, optimizer, scheduler, loss_fn, train_loader, device
        )
        train_losses.append(train_loss)
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)

        # --------- Optuna -----------
        if trial is not None:
            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # track the best score to return to Optuna
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        print(
            "========================================================================",
            flush=True,
        )
        print(
            f"Epoch {epoch} / {num_epochs}: Train loss: {train_loss} | Val loss: {val_loss}",
            flush=True,
        )

    save_training(model, train_losses, val_losses, params)

    return best_val_loss
