# ECE 285 HW2 WikiArt DCGAN

This folder contains a complete baseline workflow for auditing WikiArt, training an unconditional DCGAN, generating images, and computing IS/FID. It treats style subfolders as a single image collection; the style names are metadata and are not passed to the unconditional model.

## Files

- `config.json`: dataset path and default training settings.
- `inspect_data.py`: validates images, writes a manifest and style counts, and saves a contact sheet.
- `gan_data.py`, `gan_model.py`, `gan_utils.py`: dataset, model, and shared helpers.
- `train.py`: trains the DCGAN and saves losses, sample grids, and checkpoints.
- `generate.py`: generates individual PNGs and a grid from a checkpoint.
- `evaluate.py`: computes Ignite IS/FID against a deterministic held-out split.
- `plan.md`: project checklist and blank result sections.

## Environment setup

Use Python 3.10–3.12 for broad compatibility. Install a PyTorch and torchvision pair appropriate for the training machine first, following [the PyTorch install selector](https://pytorch.org/get-started/locally/). On a GPU machine, make sure the selected PyTorch build supports that machine's CUDA runtime. Then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

The training and generation scripts use PyTorch and Pillow; metric calculation additionally needs torchvision and PyTorch-Ignite. `requirements.txt` leaves PyTorch/torchvision version selection to the platform-specific install step.

## Run order

If a DSMLP session ends between epochs, resume from the latest saved checkpoint with `python train.py --resume outputs/checkpoints/latest.pt`. Resume restores the model, optimizers, and random-number-generator state and continues at the next epoch.

From this directory (`hw2`):

```powershell
python inspect_data.py
python train.py
python plot_losses.py
python generate.py --checkpoint outputs/checkpoints/latest.pt --count 64
python evaluate.py --checkpoint outputs/checkpoints/latest.pt --num-images 5000
```

Audit output is written to `outputs/data_audit/`. Training writes `outputs/checkpoints/`, `outputs/samples/`, `outputs/logs/losses.csv`, and the deterministic `outputs/split.json`. Evaluation metrics go to `outputs/evaluation.json`.

To use a different data location (for example, the DataHub download), change `dataset_root` in `config.json`, or make a copy such as `config_datahub.json` and pass `--config config_datahub.json` to each command. The data root should contain the style directories, such as `Impressionism/`, directly or below a parent directory.

To shorten a run or change memory use:

```powershell
python train.py --epochs 1 --batch-size 16
```

`--device auto` selects CUDA when PyTorch reports it available and otherwise uses CPU. Add `--device cuda:0` to require a particular GPU. A full 50-epoch run over the full dataset is intended for GPU execution.

## Evaluation notes

The split is reproducibly regenerated from the sorted valid image paths and the configured seed. It uses 95% for training and 5% for evaluation by default. The held-out split is used for FID; generated and real images are resized to 299×299 for Ignite's Inception metrics. Preserve the configuration, seed, metric sample count, and package versions when comparing experiments. IS and FID are useful comparisons, but should be interpreted alongside the sample grid and possible mode collapse.

## Report

Use `report_outline.md` to organize the two-page NeurIPS-format write-up. Fill in `plan.md` as results are produced. Keep `outputs/` available for plots, sample grids, checkpoints, and metrics; do not include the large raw dataset or checkpoints in the report submission unless required.
