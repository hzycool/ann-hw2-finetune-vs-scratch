from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    ResNeXt50_32X4D_Weights,
    densenet121,
    resnext50_32x4d,
)


@dataclass
class ModelBundle:
    model: nn.Module
    weights_name: str


def build_model(model_name: str, mode: str, num_classes: int) -> ModelBundle:
    if model_name == "resnext50":
        weights = ResNeXt50_32X4D_Weights.DEFAULT if mode == "finetune" else None
        model = resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        weights_name = "ImageNet1K_V2" if weights is not None else "random_init"
    elif model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if mode == "finetune" else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        weights_name = "ImageNet1K_V1" if weights is not None else "random_init"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return ModelBundle(model=model, weights_name=weights_name)


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model_name == "resnext50":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif model_name == "densenet121":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def unfreeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
