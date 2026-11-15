"""Train a DCGAN on all WikiArt style folders as one unconditional image set."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from gan_data import PaintingDataset, load_manifest, save_split, split_paths
from gan_model import Discriminator, Generator, initialize_weights
from gan_utils import load_config, save_grid, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    seed_everything(config["seed"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)

    out = config["output_dir"]
    for name in ("checkpoints", "samples", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    paths = load_manifest(config["dataset_root"], out / "data_audit" / "manifest.csv")
    train_paths, eval_paths = split_paths(paths, config["train_fraction"], config["seed"])
    save_split(config["dataset_root"], train_paths, eval_paths, out)
    dataset = PaintingDataset(train_paths, config["image_size"], config["horizontal_flip"])
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True,
        num_workers=config["num_workers"], pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers"] > 0,
    )
    if len(loader) == 0:
        raise ValueError("DataLoader has no full batches; reduce batch_size or check the dataset")

    generator = Generator(config["latent_dim"], config["image_size"], config["generator_channels"]).to(device)
    discriminator = Discriminator(config["image_size"], config["discriminator_channels"]).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["learning_rate_g"], betas=(config["adam_beta1"], config["adam_beta2"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate_d"], betas=(config["adam_beta1"], config["adam_beta2"]))
    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, config["latent_dim"], 1, 1, device=device)
    (out / "run_config.json").write_text(json.dumps({**config, "dataset_root": str(config["dataset_root"]), "output_dir": str(out), "device": str(device)}, indent=2), encoding="utf-8")
    history_path = out / "logs" / "losses.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "iteration", "loss_d", "loss_g"])
        start = time.perf_counter()
        global_step = 0
        for epoch in range(1, config["epochs"] + 1):
            generator.train()
            discriminator.train()
            total_d = total_g = 0.0
            for real in loader:
                real = real.to(device, non_blocking=True)
                batch = real.size(0)
                real_labels = torch.ones(batch, device=device)
                fake_labels = torch.zeros(batch, device=device)

                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake = generator(torch.randn(batch, config["latent_dim"], 1, 1, device=device))
                loss_d = criterion(discriminator(real), real_labels) + criterion(discriminator(fake), fake_labels)
                loss_d.backward()
                optimizer_d.step()

                optimizer_g.zero_grad(set_to_none=True)
                fake = generator(torch.randn(batch, config["latent_dim"], 1, 1, device=device))
                loss_g = criterion(discriminator(fake), real_labels)
                loss_g.backward()
                optimizer_g.step()

                global_step += 1
                total_d += loss_d.item()
                total_g += loss_g.item()
                writer.writerow([epoch, global_step, loss_d.item(), loss_g.item()])
            f.flush()
            avg_d, avg_g = total_d / len(loader), total_g / len(loader)
            elapsed = time.perf_counter() - start
            print(f"epoch {epoch:03d}/{config['epochs']} | D {avg_d:.4f} | G {avg_g:.4f} | elapsed {elapsed/60:.1f} min")
            if epoch % config["save_every_epochs"] == 0 or epoch == config["epochs"]:
                generator.eval()
                with torch.no_grad():
                    preview = generator(fixed_noise).cpu()
                save_grid(preview, out / "samples" / f"epoch_{epoch:03d}.png")
                checkpoint = {
                    "epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                    "config": {k: v for k, v in config.items() if k in {
                        "image_size", "latent_dim", "generator_channels", "discriminator_channels", "seed"
                    }},
                }
                torch.save(checkpoint, out / "checkpoints" / f"epoch_{epoch:03d}.pt")
                torch.save(checkpoint, out / "checkpoints" / "latest.pt")
    print(f"Training complete. Outputs: {out}")


if __name__ == "__main__":
    main()
