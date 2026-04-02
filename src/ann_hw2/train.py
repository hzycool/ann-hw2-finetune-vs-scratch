from __future__ import annotations

import copy
import platform
import time
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from ann_hw2.data import DatasetBundle, load_dataloaders
from ann_hw2.engine import compute_confusion, evaluate, train_one_epoch
from ann_hw2.models import build_model, count_parameters, freeze_backbone, unfreeze_all
from ann_hw2.utils import (
    dump_yaml,
    ensure_dir,
    plot_confusion_matrix,
    plot_training_curves,
    save_json,
    save_log_csv,
    save_metrics_csv,
    set_seed,
)


@dataclass
class StageConfig:
    name: str
    epochs: int
    learning_rate: float
    freeze_backbone: bool


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_stages(config: dict) -> list[StageConfig]:
    training_cfg = config["training"]
    mode = config["model"]["mode"]
    if mode == "scratch":
        return [
            StageConfig(
                name="scratch",
                epochs=int(training_cfg["scratch_epochs"]),
                learning_rate=float(training_cfg["scratch_lr"]),
                freeze_backbone=False,
            )
        ]
    if mode == "finetune":
        return [
            StageConfig(
                name="head",
                epochs=int(training_cfg["head_epochs"]),
                learning_rate=float(training_cfg["head_lr"]),
                freeze_backbone=True,
            ),
            StageConfig(
                name="full",
                epochs=int(training_cfg["full_epochs"]),
                learning_rate=float(training_cfg["full_lr"]),
                freeze_backbone=False,
            ),
        ]
    raise ValueError(f"Unsupported training mode: {mode}")


def _dataset_summary(bundle: DatasetBundle) -> dict:
    return {
        "class_names": bundle.class_names,
        "split_sizes": bundle.split_sizes,
    }


def run_experiment(config: dict) -> dict:
    training_cfg = config["training"]
    model_cfg = config["model"]
    set_seed(int(training_cfg["seed"]))

    device = _resolve_device(training_cfg["device"])
    config["training"]["resolved_device"] = str(device)

    outputs_root = ensure_dir(config["output"]["root_dir"])
    run_dir = ensure_dir(outputs_root / config["experiment_name"])
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")

    bundle = load_dataloaders(config)
    dataset_summary = _dataset_summary(bundle)

    model_bundle = build_model(
        model_name=model_cfg["name"],
        mode=model_cfg["mode"],
        num_classes=len(bundle.class_names),
    )
    model = model_bundle.model.to(device)
    criterion = nn.CrossEntropyLoss()

    stages = _build_stages(config)
    total_epochs = sum(stage.epochs for stage in stages)

    best_state = copy.deepcopy(model.state_dict())
    best_val_macro_f1 = -1.0
    best_epoch = 0
    epoch_pointer = 0
    log_rows: list[dict] = []
    train_start = time.perf_counter()

    for stage in stages:
        if stage.freeze_backbone:
            freeze_backbone(model, model_name=model_cfg["name"])
        else:
            unfreeze_all(model)

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=stage.learning_rate,
            weight_decay=float(training_cfg["weight_decay"]),
        )

        for stage_epoch in range(stage.epochs):
            train_result = train_one_epoch(
                model=model,
                dataloader=bundle.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_index=epoch_pointer,
                total_epochs=total_epochs,
            )
            val_result = evaluate(
                model=model,
                dataloader=bundle.val_loader,
                criterion=criterion,
                device=device,
                epoch_index=epoch_pointer,
                total_epochs=total_epochs,
                stage_name="val",
            )

            log_rows.append(
                {
                    "epoch": epoch_pointer + 1,
                    "stage": stage.name,
                    "stage_epoch": stage_epoch + 1,
                    "learning_rate": stage.learning_rate,
                    "train_loss": train_result.loss,
                    "train_accuracy": train_result.accuracy,
                    "train_macro_f1": train_result.macro_f1,
                    "val_loss": val_result.loss,
                    "val_accuracy": val_result.accuracy,
                    "val_macro_f1": val_result.macro_f1,
                }
            )

            if val_result.macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_result.macro_f1
                best_epoch = epoch_pointer + 1
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, checkpoint_dir / "best_model.pt")

            epoch_pointer += 1

    train_seconds = time.perf_counter() - train_start
    model.load_state_dict(best_state)

    test_result = evaluate(
        model=model,
        dataloader=bundle.test_loader,
        criterion=criterion,
        device=device,
        epoch_index=max(total_epochs - 1, 0),
        total_epochs=total_epochs,
        stage_name="test",
    )
    confusion = compute_confusion(test_result.targets, test_result.predictions)
    total_params, _ = count_parameters(model)
    unfreeze_all(model)
    _, trainable_params = count_parameters(model)

    save_log_csv(run_dir / "train_log.csv", log_rows)
    log_frame = pd.DataFrame(log_rows)
    plot_training_curves(log_frame, run_dir / "training_curves.png")
    plot_confusion_matrix(confusion, bundle.class_names, run_dir / "confusion_matrix.png")

    metrics = {
        "experiment_name": config["experiment_name"],
        "model_name": model_cfg["name"],
        "training_mode": model_cfg["mode"],
        "weights_source": model_bundle.weights_name,
        "best_epoch": best_epoch,
        "val_macro_f1_best": best_val_macro_f1,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "test_macro_f1": test_result.macro_f1,
        "train_seconds": train_seconds,
        "train_minutes": train_seconds / 60.0,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "split_train": bundle.split_sizes["train"],
        "split_validation": bundle.split_sizes["validation"],
        "split_test": bundle.split_sizes["test"],
    }
    save_metrics_csv(run_dir / "metrics.csv", metrics)
    save_json(run_dir / "dataset_summary.json", dataset_summary)
    save_json(
        run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
    )
    dump_yaml(run_dir / "config.yaml", config)

    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "dataset_summary": dataset_summary,
    }
