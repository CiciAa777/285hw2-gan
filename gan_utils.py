"""Small helpers shared by training and generation scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    base = path.resolve().parent
    for key in ("dataset_root", "output_dir"):
        value = Path(config[key])
        config[key] = value if value.is_absolute() else (base / value).resolve()
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_grid(images: torch.Tensor, path: Path, columns: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_images = [tensor_to_pil(image) for image in images]
    if not pil_images:
        return
    width, height = pil_images[0].size
    rows = (len(pil_images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(pil_images):
        grid.paste(image, ((index % columns) * width, (index // columns) * height))
    grid.save(path)


def load_generator_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    from gan_model import Generator
    config = checkpoint["config"]
    generator = Generator(config["latent_dim"], config["image_size"], config["generator_channels"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, config, checkpoint
