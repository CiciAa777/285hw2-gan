#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/private/285hw2-gan/project"
OUTPUT_DIR="outputs/trial3_batch32"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/batch_console.log") 2>&1

echo "Trial 3 GPU batch started: $(date -Is)"
echo "Project: $PWD"
echo "Dataset: $(readlink -f steubk/wikiart/versions/1)"
nvidia-smi

if [[ ! -x ../.venv_trial3/bin/python ]]; then
  python -m venv --system-site-packages ../.venv_trial3
fi
source ../.venv_trial3/bin/activate
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
PY

python -u train.py --config config_trial3.json --device cuda:0
python plot_losses.py \
  --input "$OUTPUT_DIR/logs/losses.csv" \
  --output "$OUTPUT_DIR/loss_curves.png"
python generate.py \
  --checkpoint "$OUTPUT_DIR/checkpoints/epoch_050.pt" \
  --count 64 --seed 123 \
  --output "$OUTPUT_DIR/generated"
python evaluate.py \
  --config config_trial3.json \
  --checkpoint "$OUTPUT_DIR/checkpoints/epoch_050.pt" \
  --num-images 4073 --batch-size 32 --device cuda:0
echo "Trial 3 GPU batch completed: $(date -Is)"
