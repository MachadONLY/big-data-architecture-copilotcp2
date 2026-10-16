"""Shared project utilities."""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def append_result_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id", "run_id", "timestamp", "model", "learning_rate", "epochs",
        "per_device_batch_size", "gradient_accumulation_steps", "effective_batch_size",
        "lora_rank", "lora_alpha", "lora_dropout", "max_length", "precision",
        "optimizer", "seed", "gpu", "vram_gb", "trainable_params",
        "train_loss", "validation_loss", "train_runtime_seconds",
        "max_memory_allocated_gb", "adapter_path", "run_dir", "training_log",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

def platform_summary() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
