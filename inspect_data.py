"""Audit WikiArt files and create a manifest plus a random contact sheet."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from gan_data import discover_images, verify_images, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    from gan_utils import load_config
    config = load_config(args.config)
    root, output = config["dataset_root"], config["output_dir"] / "data_audit"
    all_paths = discover_images(root)
    valid, invalid = verify_images(all_paths)
    write_manifest(root, valid, invalid, output)
    counts = Counter(path.parent.name for path in valid)
    report = {
        "dataset_root": str(root), "discovered_images": len(all_paths), "valid_images": len(valid),
        "invalid_images": len(invalid), "style_folders": len(counts),
        "images_per_style": dict(sorted(counts.items())),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rng = random.Random(config["seed"])
    chosen = rng.sample(valid, min(args.samples, len(valid)))
    tile, columns = 128, 8
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * (tile + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(chosen):
        with Image.open(path) as opened:
            image = ImageOps.fit(opened.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS)
        x, y = (i % columns) * tile, (i // columns) * (tile + 20)
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + tile + 2), path.parent.name[:20], fill="black")
    sheet.save(output / "contact_sheet.jpg", quality=90)
    print(json.dumps({k: v for k, v in report.items() if k != "images_per_style"}, indent=2))
    print(f"Audit files: {output}")


if __name__ == "__main__":
    main()
