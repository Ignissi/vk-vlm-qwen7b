#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Не найден $PYTHON_BIN. На Ubuntu 24.04 нужен Python 3.12."
  exit 1
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# RTX 5090 (Blackwell, sm_120) требует сборку PyTorch с CUDA 12.8 или новее.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python check_gpu.py

echo
echo "Окружение готово. Следующая команда: ./run.sh"
