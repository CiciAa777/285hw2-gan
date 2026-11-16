#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/private/285hw2-gan/project"
OUTPUT_DIR="outputs/trial3_batch32"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/batch_console.log") 2>&1

echo "Trial 3 GPU batch started: $(date -Is)"
echo "Project: $PWD"
nvidia-smi

RUNTIME_CONFIG="$OUTPUT_DIR/gpu_runtime_config.json"
python - "$RUNTIME_CONFIG" config_trial3.json "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

destination, source, output_dir = sys.argv[1:]
config = json.loads(Path(source).read_text(encoding="utf-8"))
project = Path.home() / "private/285hw2-gan/project"
config["dataset_root"] = str(Path.home() / "private/285hw2-gan/wikiart/versions/1")
config["output_dir"] = str(project / output_dir)
if not Path(config["dataset_root"]).is_dir():
    raise SystemExit(f"Dataset directory not found: {config['dataset_root']}")
Path(destination).write_text(json.dumps(config, indent=2), encoding="utf-8")
print("Dataset:", config["dataset_root"])
print("Output:", config["output_dir"])
PY

if [[ ! -x ../.venv_trial3/bin/python ]]; then
  python -m venv --system-site-packages ../.venv_trial3
fi
source ../.venv_trial3/bin/activate
# Use the image's PyTorch 2.2/CUDA 12.1 build, which supports the small
# Pascal-class GTX 1080 Ti; ignore the newer user-site PyTorch 2.9 build.
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt

python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this pod; stopping before training")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
torch.randn(64, device="cuda").normal_()
torch.cuda.synchronize()
print("CUDA kernel smoke check: passed")
PY

python -u train.py --config "$RUNTIME_CONFIG" --device cuda:0
python plot_losses.py \
  --input "$OUTPUT_DIR/logs/losses.csv" \
  --output "$OUTPUT_DIR/loss_curves.png"
python generate.py \
  --checkpoint "$OUTPUT_DIR/checkpoints/epoch_050.pt" \
  --count 64 --seed 123 \
  --output "$OUTPUT_DIR/generated"
python evaluate.py \
  --config "$RUNTIME_CONFIG" \
  --checkpoint "$OUTPUT_DIR/checkpoints/epoch_050.pt" \
  --num-images 4073 --batch-size 32 --device cuda:0
echo "Trial 3 GPU batch completed: $(date -Is)"
