"""Plot per-step GAN losses and their moving averages from the training CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "outputs" / "logs" / "losses.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "plots" / "loss_curves.png")
    parser.add_argument("--window", type=int, default=250, help="moving-average window in training steps")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No loss rows found in {args.input}")
    steps = np.asarray([int(row["iteration"]) for row in rows])
    loss_d = np.asarray([float(row["loss_d"]) for row in rows])
    loss_g = np.asarray([float(row["loss_g"]) for row in rows])
    window = min(max(1, args.window), len(rows))
    kernel = np.ones(window, dtype=np.float64) / window
    smooth_d = np.convolve(loss_d, kernel, mode="valid")
    smooth_g = np.convolve(loss_g, kernel, mode="valid")
    smooth_steps = steps[window - 1:]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, loss_d, color="tab:blue", alpha=0.18, linewidth=0.5, label="D loss (per step)")
    ax.plot(steps, loss_g, color="tab:orange", alpha=0.18, linewidth=0.5, label="G loss (per step)")
    ax.plot(smooth_steps, smooth_d, color="tab:blue", linewidth=1.5, label=f"D moving avg ({window})")
    ax.plot(smooth_steps, smooth_g, color="tab:orange", linewidth=1.5, label=f"G moving avg ({window})")
    ax.set(xlabel="Training step", ylabel="Binary cross-entropy loss", title="DCGAN training losses")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved loss plot: {args.output}")


if __name__ == "__main__":
    main()
