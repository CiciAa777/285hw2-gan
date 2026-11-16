#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/private/285hw2-gan/project"
OUTPUT_DIR="outputs/trial2_lr_d_5e5"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/cpu_batch_console.log") 2>&1

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Trial 2 CPU batch started: $(date -Is)"
echo "Project: $PWD"
echo "CPU allocation: ${KUBERNETES_LIMIT_CPU:-unknown}"

RUNTIME_CONFIG="$OUTPUT_DIR/cpu_runtime_config.json"
python - "$RUNTIME_CONFIG" config_trial2.json "$OUTPUT_DIR" <<'PY'
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

if [[ ! -x ../.venv_trial2_cpu/bin/python ]]; then
  python -m venv --system-site-packages ../.venv_trial2_cpu
fi
source ../.venv_trial2_cpu/bin/activate
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Training device: CPU")
PY

python -u train.py --config "$RUNTIME_CONFIG" --device cpu
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
  --num-images 4073 --batch-size 32 --device cpu
echo "Trial 2 CPU batch completed: $(date -Is)"
