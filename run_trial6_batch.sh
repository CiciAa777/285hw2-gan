#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="$HOME/private/285hw2-gan/wikiart/versions/1"
cd "$PROJECT"

if [[ ! -d "$DATASET" ]]; then
  echo "Dataset directory not found: $DATASET" >&2
  exit 2
fi
if [[ ! -x ../.venv_trial5_gpu/bin/python ]]; then
  echo "Expected CUDA-enabled venv is missing: $PROJECT/../.venv_trial5_gpu" >&2
  exit 2
fi
source ../.venv_trial5_gpu/bin/activate

python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; refusing to train on CPU")
print("GPU:", torch.cuda.get_device_name(0))
torch.randn(64, device="cuda").normal_()
torch.cuda.synchronize()
print("CUDA smoke check: passed")
PY

OUTPUT="outputs/trial6_g_width96"
mkdir -p "$OUTPUT"
RUNTIME_CONFIG="$OUTPUT/gpu_runtime_config.json"
python - "$RUNTIME_CONFIG" config_trial6_g_width96.json "$OUTPUT" "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

destination, source, relative_output, dataset = sys.argv[1:]
cfg = json.loads(Path(source).read_text(encoding="utf-8"))
cfg["dataset_root"] = dataset
cfg["output_dir"] = str(Path.cwd() / relative_output)
cfg["save_every_epochs"] = 25
Path(destination).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("Config:", destination)
print("Generator channels:", cfg["generator_channels"])
PY

python -u train.py --config "$RUNTIME_CONFIG" --device cuda:0
python plot_losses.py --input "$OUTPUT/logs/losses.csv" --output "$OUTPUT/loss_curves.png"
python generate.py --checkpoint "$OUTPUT/checkpoints/epoch_050.pt" --count 64 --seed 123 --output "$OUTPUT/generated"
python evaluate.py --config "$RUNTIME_CONFIG" --checkpoint "$OUTPUT/checkpoints/epoch_050.pt" --num-images 4073 --batch-size 32 --device cuda:0
echo "Trial 6 medium-GPU batch completed: $(date -Is)"
