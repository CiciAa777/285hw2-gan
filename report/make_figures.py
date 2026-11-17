"""Create the report's epoch-mean training-loss figure from archived logs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures" / "trial2_losses.png"
csv_path = ROOT / "results" / "trial2_lr_d_5e5" / "losses.csv"
data = pd.read_csv(csv_path).groupby("epoch", as_index=False)[["loss_d", "loss_g"]].mean()

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9})
fig, ax = plt.subplots(figsize=(5.6, 2.25), constrained_layout=True)
ax.plot(data["epoch"], data["loss_d"], color="#3568a8", lw=1.7, label="Discriminator")
ax.plot(data["epoch"], data["loss_g"], color="#c45a32", lw=1.7, label="Generator")
ax.set(xlabel="Epoch", ylabel="Mean BCE loss per epoch", xlim=(1, 50))
ax.grid(True, alpha=0.25, linewidth=0.6)
ax.legend(frameon=False, ncol=2, loc="upper left")
fig.savefig(OUT, dpi=240, bbox_inches="tight")
print(OUT)
