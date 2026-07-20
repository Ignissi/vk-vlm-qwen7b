from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass(frozen=True)
class RunConfig:
    model_id: str
    dataset_id: str
    dataset_split: str
    train_size: int
    eval_size: int
    epochs: float
    seed: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    gradient_accumulation_steps: int
    min_pixels: int
    max_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA-адаптация Qwen2.5-VL-7B на DeepVK MMBench-ru")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset-id", default="deepvk/MMBench-ru")
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def ensure_rgb(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Ожидалось PIL.Image, получено: {type(image)!r}")
    return image.convert("RGB")


def available_options(example: dict[str, Any]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for letter in ("A", "B", "C", "D"):
        value = clean_value(example.get(letter))
        if value:
            options.append((letter, value))
    return options


def is_valid_example(example: dict[str, Any]) -> bool:
    try:
        ensure_rgb(example.get("image"))
    except Exception:
        return False
    question = clean_value(example.get("question"))
    answer = clean_value(example.get("answer")).upper()
    option_letters = {letter for letter, _ in available_options(example)}
    return bool(question and len(option_letters) >= 2 and answer in option_letters)


def build_question(example: dict[str, Any]) -> str:
    question = clean_value(example["question"])
    hint = clean_value(example.get("hint"))
    options = available_options(example)

    parts = []
    if hint:
        parts.append(f"Контекст: {hint}")
    parts.append(f"Вопрос: {question}")
    parts.append("Варианты ответа:")
    parts.extend(f"{letter}. {text}" for letter, text in options)
    parts.append("Ответь только одной буквой правильного варианта: A, B, C или D.")
    return "\n".join(parts)


def select_valid_examples(raw: Dataset, required: int, seed: int) -> Dataset:
    shuffled = raw.shuffle(seed=seed)
    selected_indices: list[int] = []
    for idx in range(len(shuffled)):
        try:
            example = shuffled[idx]
            if is_valid_example(example):
                selected_indices.append(idx)
        except Exception as exc:
            print(f"Пропущена повреждённая запись {idx}: {exc}")
        if len(selected_indices) == required:
            break
    if len(selected_indices) < required:
        raise RuntimeError(f"Найдено только {len(selected_indices)} валидных записей из требуемых {required}.")
    return shuffled.select(selected_indices)


class QwenVLCollator:
    def __init__(self, processor: AutoProcessor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[Image.Image] = []

        for example in features:
            image = ensure_rgb(example["image"])
            question = build_question(example)
            answer = clean_value(example["answer"]).upper()

            user_message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
            full_messages = [
                user_message,
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]

            prompt_texts.append(
                self.processor.apply_chat_template(
                    [user_message], tokenize=False, add_generation_prompt=True
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(image)

        full_batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100

        for row in range(len(features)):
            prompt_length = int(prompt_batch["attention_mask"][row].sum().item())
            labels[row, :prompt_length] = -100

        full_batch["labels"] = labels
        return dict(full_batch)


def extract_choice(text: str) -> str:
    upper = text.upper().strip()
    match = re.search(r"(?:^|[^A-ZА-Я])([ABCD])(?:$|[^A-ZА-Я])", upper)
    if match:
        return match.group(1)
    if upper and upper[0] in "ABCD":
        return upper[0]
    return ""


@torch.inference_mode()
def predict_one(
    model: torch.nn.Module,
    processor: AutoProcessor,
    example: dict[str, Any],
) -> tuple[str, str]:
    image = ensure_rgb(example["image"])
    message = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": build_question(example)},
        ],
    }
    text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        use_cache=True,
    )
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return extract_choice(decoded), decoded


def evaluate_model(
    model: torch.nn.Module,
    processor: AutoProcessor,
    dataset: Dataset,
    stage: str,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    correct = 0

    print(f"\nОценка: {stage} ({len(dataset)} примеров)")
    for idx in range(len(dataset)):
        example = dataset[idx]
        expected = clean_value(example["answer"]).upper()
        predicted, raw_output = predict_one(model, processor, example)
        is_correct = predicted == expected
        correct += int(is_correct)
        rows.append(
            {
                "sample_index": clean_value(example.get("index")) or idx,
                "question": clean_value(example["question"]),
                "expected": expected,
                f"{stage}_prediction": predicted,
                f"{stage}_raw_output": raw_output,
                f"{stage}_correct": is_correct,
            }
        )
        print(f"  {idx + 1:02d}/{len(dataset)}: expected={expected}, predicted={predicted or '∅'}")

    accuracy = correct / len(dataset)
    print(f"Accuracy {stage}: {accuracy:.4f}")
    return accuracy, rows


def merge_prediction_rows(
    base_rows: list[dict[str, Any]], adapted_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for base, adapted in zip(base_rows, adapted_rows, strict=True):
        row = dict(base)
        row.update(
            {
                key: value
                for key, value in adapted.items()
                if key.startswith("adapted_")
            }
        )
        merged.append(row)
    return merged


def write_report(
    path: Path,
    config: RunConfig,
    base_accuracy: float,
    adapted_accuracy: float,
    train_metrics: dict[str, Any],
    elapsed_minutes: float,
) -> None:
    delta = adapted_accuracy - base_accuracy
    direction = "выросла" if delta > 0 else "снизилась" if delta < 0 else "не изменилась"
    report = f"""# Отчёт по проекту

## 1. Название

LoRA-адаптация визуально-языковой модели для русскоязычного анализа изображений на данных DeepVK.

## 2. Смысл и цель

Изображение содержит сведения, которые обычная текстовая поисковая система не может извлечь напрямую. Визуально-языковая модель позволяет задать вопрос об изображении на русском языке и получить ответ, основанный на его содержимом.

**Цель:** адаптировать модель `{config.model_id}` к русскоязычным вопросам по изображениям с использованием открытого датасета `{config.dataset_id}` и измерить изменение точности после дообучения.

## 3. Задачи

1. Подготовить непересекающиеся обучающую и контрольную выборки.
2. Измерить точность исходной модели.
3. Выполнить параметрически эффективное дообучение методом LoRA.
4. Сохранить обученный адаптер.
5. Повторно измерить точность и сравнить результаты.

## 4. Использование открытых данных VK

Использован датасет `{config.dataset_id}`, опубликованный DeepVK. Каждая запись содержит изображение, вопрос на русском языке, варианты ответа и правильную букву. Из единственного split `{config.dataset_split}` после перемешивания с `seed={config.seed}` выделено {config.train_size} примеров для обучения и {config.eval_size} других примеров для внутренней контрольной оценки.

Данное разделение используется только для учебного эксперимента. Полученная метрика не объявляется официальным результатом полного MMBench.

## 5. Модель и обучение

- базовая модель: `{config.model_id}`;
- точность вычислений: BF16;
- метод: LoRA;
- обучаемые проекции: `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- LoRA rank: {config.lora_rank};
- LoRA alpha: {config.lora_alpha};
- эпохи: {config.epochs};
- batch size: 1;
- gradient accumulation: {config.gradient_accumulation_steps};
- learning rate: {config.learning_rate};
- визуальное разрешение ограничено диапазоном {config.min_pixels}–{config.max_pixels} пикселей для снижения вычислительной нагрузки.

Исходные веса модели не изменялись. Обучались только параметры LoRA-адаптера.

## 6. Результаты

| Модель | Accuracy |
|---|---:|
| Исходная | {base_accuracy:.4f} |
| После LoRA | {adapted_accuracy:.4f} |
| Изменение | {delta:+.4f} |

После обучения точность {direction}. Это является измеренным результатом эксперимента; значения не корректировались вручную.

Время полного запуска: приблизительно {elapsed_minutes:.1f} мин.

Train loss: {train_metrics.get('train_loss', 'не зафиксирован')}.

## 7. Полученные материалы

- `adapter/` — обученные LoRA-веса и конфигурация процессора;
- `results/metrics.json` — итоговые метрики;
- `results/predictions.csv` — ответы до и после обучения;
- `results/run_config.json` — параметры эксперимента;
- `train_and_evaluate.py` — воспроизводимый код.

## 8. Вывод

Реализован полный цикл адаптации визуально-языковой модели: открытые данные DeepVK были подготовлены, исходная модель оценена, LoRA-адаптер обучен и сохранён, после чего проведена повторная оценка на непересекающейся контрольной выборке.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    adapter_dir = output_dir / "adapter"
    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна. Сначала выполни ./setup.sh и проверь драйвер.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 недоступен в текущей установке PyTorch.")

    set_seed(args.seed)
    random.seed(args.seed)
    started = time.time()

    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    config = RunConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        train_size=args.train_size,
        eval_size=args.eval_size,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
        lora_rank=8,
        lora_alpha=16,
        gradient_accumulation_steps=4,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    (results_dir / "run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Загрузка датасета DeepVK...")
    raw = load_dataset(args.dataset_id, split=args.dataset_split)
    selected = select_valid_examples(raw, args.train_size + args.eval_size, args.seed)
    train_dataset = selected.select(range(args.train_size))
    eval_dataset = selected.select(range(args.train_size, args.train_size + args.eval_size))

    print("Загрузка процессора и Qwen2.5-VL-7B в BF16...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    processor.tokenizer.padding_side = "right"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to("cuda")

    base_accuracy, base_rows = evaluate_model(model, processor, eval_dataset, "base")

    print("\nПодключение LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=4,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QwenVLCollator(processor),
    )
    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)

    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)

    model.config.use_cache = True
    adapted_accuracy, adapted_rows = evaluate_model(model, processor, eval_dataset, "adapted")
    predictions = merge_prediction_rows(base_rows, adapted_rows)

    delta = adapted_accuracy - base_accuracy
    metrics = {
        "base_accuracy": base_accuracy,
        "adapted_accuracy": adapted_accuracy,
        "accuracy_delta": delta,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "train_metrics": train_metrics,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(predictions).to_csv(results_dir / "predictions.csv", index=False)

    elapsed_minutes = (time.time() - started) / 60
    write_report(
        output_dir / "REPORT.md",
        config,
        base_accuracy,
        adapted_accuracy,
        train_metrics,
        elapsed_minutes,
    )

    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== ГОТОВО ===")
    print(f"Исходная accuracy: {base_accuracy:.4f}")
    print(f"После LoRA:        {adapted_accuracy:.4f}")
    print(f"Изменение:         {delta:+.4f}")
    print(f"Адаптер:           {adapter_dir}")
    print(f"Отчёт:             {output_dir / 'REPORT.md'}")
    print(f"Результаты:        {results_dir}")


if __name__ == "__main__":
    main()
