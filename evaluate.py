"""Compute Ignite Inception Score and FID against the held-out real split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from gan_data import PaintingDataset, load_manifest, split_paths
from gan_utils import load_config, load_generator_checkpoint


def to_metric_resolution(batch: torch.Tensor, size: int = 299) -> torch.Tensor:
    # Match the Ignite tutorial's PIL-based resize and convert from model [-1,1] to metric [0,1].
    images = []
    for tensor in batch:
        array = tensor.detach().cpu().clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).numpy()
        image = Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.BILINEAR)
        images.append(torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "outputs" / "checkpoints" / "latest.pt")
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        from ignite.metrics import FID, InceptionScore
    except ImportError as exc:
        raise SystemExit("Install pytorch-ignite and a compatible torchvision build to compute IS/FID") from exc
    config = load_config(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    generator, checkpoint_config, checkpoint = load_generator_checkpoint(args.checkpoint, device)
    paths = load_manifest(config["dataset_root"], config["output_dir"] / "data_audit" / "manifest.csv")
    _, evaluation_paths = split_paths(paths, config["train_fraction"], config["seed"])
    n = min(args.num_images, len(evaluation_paths))
    if n < 2:
        raise ValueError("At least two held-out images are needed for FID")
    real_loader = DataLoader(PaintingDataset(evaluation_paths[:n], checkpoint_config["image_size"]), batch_size=args.batch_size, shuffle=False, num_workers=config["num_workers"])
    fid = FID(device=device)
    inception_score = InceptionScore(device=device)
    remaining = n
    generator.eval()
    with torch.no_grad():
        for real in real_loader:
            batch_size = real.size(0)
            fake = generator(torch.randn(batch_size, checkpoint_config["latent_dim"], 1, 1, device=device)).cpu()
            real_metric, fake_metric = to_metric_resolution(real), to_metric_resolution(fake)
            fid.update((fake_metric, real_metric))
            inception_score.update(fake_metric)
            remaining -= batch_size
            if remaining <= 0:
                break
    result = {
        "checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"], "num_real": n, "num_generated": n,
        "image_size": checkpoint_config["image_size"], "metric_input_size": 299,
        "fid": float(fid.compute()), "inception_score": inception_score.compute(),
        "ignite_version": __import__("ignite").__version__, "device": str(device),
    }
    score = result["inception_score"]
    if isinstance(score, tuple):
        result["inception_score"] = [float(v) for v in score]
    else:
        result["inception_score"] = float(score)
    output = config["output_dir"] / "evaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
