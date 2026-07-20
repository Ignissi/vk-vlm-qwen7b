#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_and_evaluate.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --train-size 512 \
  --eval-size 128 \
  --epochs 2 \
  --learning-rate 5e-5 \
  --output-dir .
