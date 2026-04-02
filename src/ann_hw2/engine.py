from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float
    predictions: list[int]
    targets: list[int]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
) -> EpochResult:
    model.train()
    running_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch_index + 1}/{total_epochs} [train]",
        leave=False,
    )
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    epoch_macro_f1 = f1_score(all_targets, all_predictions, average="macro")
    return EpochResult(
        loss=epoch_loss,
        accuracy=epoch_accuracy,
        macro_f1=epoch_macro_f1,
        predictions=all_predictions,
        targets=all_targets,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
    stage_name: str,
) -> EpochResult:
    model.eval()
    running_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch_index + 1}/{total_epochs} [{stage_name}]",
        leave=False,
    )
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    epoch_macro_f1 = f1_score(all_targets, all_predictions, average="macro")
    return EpochResult(
        loss=epoch_loss,
        accuracy=epoch_accuracy,
        macro_f1=epoch_macro_f1,
        predictions=all_predictions,
        targets=all_targets,
    )


def compute_confusion(targets: list[int], predictions: list[int]) -> np.ndarray:
    return confusion_matrix(targets, predictions)
