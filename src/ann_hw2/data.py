from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    split_sizes: dict[str, int]


class BeansTorchDataset(Dataset):
    def __init__(self, split, transform: Callable | None = None) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.split[index]
        image: Image.Image = item["image"].convert("RGB")
        label = int(item["labels"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _build_transforms(config: dict, train: bool) -> transforms.Compose:
    image_size = int(config["image_size"])
    mean = config["mean"]
    std = config["std"]
    if train:
        color_jitter_cfg = config["train_aug"]["color_jitter"]
        transform_steps = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(
                p=float(config["train_aug"]["horizontal_flip"])
            ),
            transforms.ColorJitter(
                brightness=float(color_jitter_cfg["brightness"]),
                contrast=float(color_jitter_cfg["contrast"]),
                saturation=float(color_jitter_cfg["saturation"]),
                hue=float(color_jitter_cfg["hue"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        resize_size = int(round(image_size * 256 / 224))
        transform_steps = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    return transforms.Compose(transform_steps)


def _load_beans_dataset(cache_dir: str | Path) -> DatasetDict:
    return load_dataset("beans", cache_dir=str(cache_dir))


def load_dataloaders(config: dict) -> DatasetBundle:
    dataset_cfg = config["dataset"]
    cache_dir = Path(dataset_cfg["cache_dir"])
    raw_dataset = _load_beans_dataset(cache_dir=cache_dir)
    label_names = list(raw_dataset["train"].features["labels"].names)

    train_dataset = BeansTorchDataset(
        raw_dataset["train"], transform=_build_transforms(dataset_cfg, train=True)
    )
    val_dataset = BeansTorchDataset(
        raw_dataset["validation"], transform=_build_transforms(dataset_cfg, train=False)
    )
    test_dataset = BeansTorchDataset(
        raw_dataset["test"], transform=_build_transforms(dataset_cfg, train=False)
    )

    batch_size = int(dataset_cfg["batch_size"])
    num_workers = int(dataset_cfg["num_workers"])
    pin_memory = config["training"]["device"] != "cpu"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=label_names,
        split_sizes={
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
        },
    )
