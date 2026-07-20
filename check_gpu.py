from __future__ import annotations

import sys
import torch


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна. Проверь драйвер NVIDIA и установку PyTorch cu128.")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name}")
    print(f"Compute capability: sm_{capability[0]}{capability[1]}")
    print(f"VRAM: {total_gb:.1f} GB")
    print(f"BF16: {torch.cuda.is_bf16_supported()}")

    if capability < (12, 0):
        print("Предупреждение: ожидалась RTX 5090 / sm_120, но запуск всё равно возможен.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("GPU/PyTorch не сообщает поддержку BF16; этот проект настроен на BF16.")


if __name__ == "__main__":
    main()
