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
echo "Dataset: $(readlink -f steubk/wikiart/versions/1)"
echo "CPU allocation: ${KUBERNETES_LIMIT_CPU:-unknown}"

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

python -u train.py --config config_trial2.json --device cpu
python plot_losses.py \
  --input "$OUTPUT_DIR/logs/losses.csv" \
  --output "$OUTPUT_DIR/loss_curves.png"
python generate.py \
  --checkpoint "$OUTPUT_DIR/checkpoints/epoch_050.pt" \
  --count 64 --seed 123 \
  --output "$OUTPUT_DIR/generated"
python evaluate.py \
  --config config_trial2.json \
  --checkpoint "$OUTPUT_DIR/checkpoints/epoch_050.pt" \
  --num-images 4073 --batch-size 32 --device cpu
echo "Trial 2 CPU batch completed: $(date -Is)"
