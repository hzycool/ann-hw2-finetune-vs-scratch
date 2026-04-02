from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def merge_dicts(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def save_log_csv(path: str | Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def save_metrics_csv(path: str | Path, metrics: dict) -> None:
    frame = pd.DataFrame([metrics])
    frame.to_csv(path, index=False)


def save_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def plot_training_curves(log_frame: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    axes[0].plot(log_frame["epoch"], log_frame["train_loss"], label="train")
    axes[0].plot(log_frame["epoch"], log_frame["val_loss"], label="validation")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].legend()

    axes[1].plot(log_frame["epoch"], log_frame["train_accuracy"], label="train acc")
    axes[1].plot(log_frame["epoch"], log_frame["val_accuracy"], label="val acc")
    axes[1].plot(log_frame["epoch"], log_frame["val_macro_f1"], label="val macro-F1")
    axes[1].set_title("Accuracy / Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.3, linestyle="--")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: Iterable[str],
    output_path: str | Path,
) -> None:
    labels = list(class_names)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=220)
    image = ax.imshow(confusion, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    threshold = confusion.max() / 2.0 if confusion.size else 0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            color = "white" if confusion[row, col] > threshold else "black"
            ax.text(
                col,
                row,
                str(confusion[row, col]),
                ha="center",
                va="center",
                color=color,
            )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_table(path: str | Path, lines: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
