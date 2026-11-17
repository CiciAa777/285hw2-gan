"""Train a DCGAN on all WikiArt style folders as one unconditional image set."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
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
    parser.add_argument("--resume", type=Path, help="resume from an epoch checkpoint")
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
    generator_updates_per_discriminator = int(config.get("generator_updates_per_discriminator", 1))
    if generator_updates_per_discriminator < 1:
        raise ValueError("generator_updates_per_discriminator must be at least 1")

    generator = Generator(
        config["latent_dim"], config["image_size"], config["generator_channels"],
        config.get("generator_activation", "relu"),
    ).to(device)
    discriminator = Discriminator(
        config["image_size"], config["discriminator_channels"],
        spectral_norm=config.get("spectral_norm_discriminator", False),
    ).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["learning_rate_g"], betas=(config["adam_beta1"], config["adam_beta2"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate_d"], betas=(config["adam_beta1"], config["adam_beta2"]))
    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, config["latent_dim"], 1, 1, device=device)
    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        rng = checkpoint.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        if checkpoint.get("fixed_noise") is not None:
            fixed_noise = checkpoint["fixed_noise"].to(device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)
    (out / "run_config.json").write_text(json.dumps({**config, "dataset_root": str(config["dataset_root"]), "output_dir": str(out), "device": str(device)}, indent=2), encoding="utf-8")
    history_path = out / "logs" / "losses.csv"
    log_mode = "a" if args.resume is not None and history_path.exists() else "w"
    with history_path.open(log_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if log_mode == "w" or history_path.stat().st_size == 0:
            writer.writerow(["epoch", "iteration", "loss_d", "loss_g"])
        start = time.perf_counter()
        for epoch in range(start_epoch, config["epochs"] + 1):
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

                loss_g_total = 0.0
                for _ in range(generator_updates_per_discriminator):
                    optimizer_g.zero_grad(set_to_none=True)
                    fake = generator(torch.randn(batch, config["latent_dim"], 1, 1, device=device))
                    loss_g = criterion(discriminator(fake), real_labels)
                    loss_g.backward()
                    optimizer_g.step()
                    loss_g_total += loss_g.item()
                mean_loss_g = loss_g_total / generator_updates_per_discriminator

                global_step += 1
                total_d += loss_d.item()
                total_g += mean_loss_g
                writer.writerow([epoch, global_step, loss_d.item(), mean_loss_g])
            f.flush()
            avg_d, avg_g = total_d / len(loader), total_g / len(loader)
            elapsed = time.perf_counter() - start
            print(f"epoch {epoch:03d}/{config['epochs']} | D {avg_d:.4f} | G {avg_g:.4f} | elapsed {elapsed/60:.1f} min", flush=True)
            if epoch % config["save_every_epochs"] == 0 or epoch == config["epochs"]:
                generator.eval()
                with torch.no_grad():
                    preview = generator(fixed_noise).cpu()
                save_grid(preview, out / "samples" / f"epoch_{epoch:03d}.png")
                checkpoint = {
                    "epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                    "global_step": global_step,
                    "fixed_noise": fixed_noise.detach().cpu(),
                    "rng_state": {
                        "python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                    "config": {k: v for k, v in config.items() if k in {
                        "image_size", "latent_dim", "generator_channels", "discriminator_channels",
                        "spectral_norm_discriminator", "generator_activation", "seed"
                    }},
                }
                torch.save(checkpoint, out / "checkpoints" / f"epoch_{epoch:03d}.pt")
                torch.save(checkpoint, out / "checkpoints" / "latest.pt")
    print(f"Training complete. Outputs: {out}")


if __name__ == "__main__":
    main()
