"""Generate paintings from a trained generator checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gan_utils import load_generator_checkpoint, save_grid, seed_everything, tensor_to_pil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "outputs" / "checkpoints" / "latest.pt")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    seed_everything(args.seed)
    generator, config, checkpoint = load_generator_checkpoint(args.checkpoint, device)
    output = args.output or args.checkpoint.parent.parent / "generated" / f"epoch_{checkpoint['epoch']:03d}"
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    with torch.no_grad():
        for start in range(0, args.count, 64):
            batch = min(64, args.count - start)
            noise = torch.randn(batch, config["latent_dim"], 1, 1, device=device)
            generated.extend(generator(noise).cpu().unbind(0))
    for index, image in enumerate(generated):
        tensor_to_pil(image).save(output / f"sample_{index:05d}.png")
    save_grid(torch.stack(generated[:min(64, len(generated))]), output / "grid.png")
    print(f"Saved {len(generated)} images to {output}")


if __name__ == "__main__":
    main()
