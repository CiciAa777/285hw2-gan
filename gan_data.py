"""Dataset discovery, deterministic splitting, and image preprocessing."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def discover_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def verify_images(paths: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    valid, invalid = [], []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append(path)
        except Exception as exc:
            invalid.append((path, str(exc)))
    return valid, invalid


def write_manifest(root: Path, valid: list[Path], invalid: list[tuple[Path, str]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "style"])
        for path in valid:
            writer.writerow([path.relative_to(root).as_posix(), path.parent.name])
    with (output / "invalid_images.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "error"])
        writer.writerows((str(path), error) for path, error in invalid)


def load_manifest(root: Path, manifest_path: Path | None) -> list[Path]:
    if manifest_path is not None and manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            return [root / row["relative_path"] for row in csv.DictReader(f)]
    return discover_images(root)


def split_paths(paths: list[Path], train_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    if len(paths) < 2:
        raise ValueError(f"Need at least two valid images, found {len(paths)}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    indices = list(range(len(paths)))
    random.Random(seed).shuffle(indices)
    n_train = min(max(1, int(len(paths) * train_fraction)), len(paths) - 1)
    train = [paths[i] for i in indices[:n_train]]
    evaluation = [paths[i] for i in indices[n_train:]]
    return train, evaluation


def save_split(root: Path, train: list[Path], evaluation: list[Path], output: Path) -> None:
    with (output / "split.json").open("w", encoding="utf-8") as f:
        json.dump({
            "train": [p.relative_to(root).as_posix() for p in train],
            "evaluation": [p.relative_to(root).as_posix() for p in evaluation],
        }, f, indent=2)


class PaintingDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int, horizontal_flip: bool = False):
        self.paths = paths
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        with Image.open(path) as opened:
            image = ImageOps.fit(opened.convert("RGB"), (self.image_size, self.image_size), method=Image.Resampling.LANCZOS)
        if self.horizontal_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1)
