"""QLoRA supervised fine-tuning for the Big Data Architecture Copilot.

This script intentionally refuses to train without CUDA because the course requirement
asks for GPU-based experiments. It NEVER reads data/processed/test.jsonl.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.metadata as metadata
import json
import shutil
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from common import ROOT, append_result_csv, ensure_dir, load_yaml, write_json
from gpu_check import collect as collect_gpu_info

TRAIN_FILE = ROOT / "data" / "processed" / "train.jsonl"
VAL_FILE = ROOT / "data" / "processed" / "validation.jsonl"


class _Tee:
    """Mirror console output to an evidence log without hiding live progress."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, data):
        self.console.write(data)
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return bool(getattr(self.console, "isatty", lambda: False)())


def enable_training_log(path: Path) -> None:
    handle = path.open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, handle)
    sys.stderr = _Tee(original_stderr, handle)

    def restore() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            handle.close()

    atexit.register(restore)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in PyTorch. Training aborted because the project "
            "requires the main experiments to run on GPU."
        )


def package_versions() -> dict[str, str]:
    packages = [
        "torch", "transformers", "trl", "peft", "datasets",
        "accelerate", "bitsandbytes", "safetensors", "pyyaml",
    ]
    out = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def choose_precision() -> tuple[torch.dtype, str, bool, bool]:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16", True, False
    return torch.float16, "fp16", False, True


def load_prompt_completion_dataset(tokenizer):
    data = load_dataset(
        "json",
        data_files={"train": str(TRAIN_FILE), "validation": str(VAL_FILE)},
    )

    def add_eos(example):
        completion = example["completion"]
        if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
            completion = completion + tokenizer.eos_token
        return {"completion": completion}

    data = data.map(add_eos, num_proc=1)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--model-override",
        default=None,
        help="Optional fallback, e.g. Qwen/Qwen3-0.6B-Base if the GPU cannot fit 1.7B.",
    )
    args = parser.parse_args()

    require_cuda()
    cfg = load_yaml(args.config)
    if args.model_override:
        cfg["model_name"] = args.model_override

    set_seed(int(cfg["seed"]))
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    experiment_id = cfg["experiment_id"]
    exp_dir = ensure_dir(ROOT / "experiments" / experiment_id)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ensure_dir(exp_dir / "runs" / run_id)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    adapter_dir = run_dir / "adapter"
    training_log = run_dir / "training.log"
    enable_training_log(training_log)

    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    gpu_info = collect_gpu_info()
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "packages": package_versions(),
        "gpu": gpu_info,
        "config": cfg,
        "training_data": {
            "train_file": str(TRAIN_FILE.relative_to(ROOT)),
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_file": str(VAL_FILE.relative_to(ROOT)),
            "validation_sha256": sha256_file(VAL_FILE),
            "test_file_loaded": False,
        },
    }
    write_json(run_dir / "environment.json", environment)

    compute_dtype, precision_name, use_bf16, use_fp16 = choose_precision()
    print(f"Precision selected: {precision_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {cfg['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_config = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    dataset = load_prompt_completion_dataset(tokenizer)

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        optim=str(cfg["optimizer"]),
        max_length=int(cfg["max_length"]),
        completion_only_loss=True,
        packing=False,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        dataset_num_proc=1,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("\nTRAINING STARTED - open another VSCode terminal and run nvidia-smi now.")
    train_result = trainer.train()
    validation = trainer.evaluate()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()

    # Keep a stable "latest adapter" path for the app while preserving timestamped runs.
    latest_adapter = exp_dir / "adapter"
    if latest_adapter.exists():
        shutil.rmtree(latest_adapter)
    shutil.copytree(adapter_dir, latest_adapter)

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    max_alloc = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    metrics = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "model": cfg["model_name"],
        "precision": precision_name,
        "trainable_params": trainable,
        "total_params_seen_by_peft": total,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation,
        "max_memory_allocated_gb": round(max_alloc, 3),
        "total_vram_gb": round(total_vram, 3),
        "gpu": torch.cuda.get_device_name(0),
        "adapter_path": str(latest_adapter),
        "training_log": str(training_log),
        "important_note": "No test-set example was loaded by train.py.",
    }
    write_json(run_dir / "metrics.json", metrics)

    result_row = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": cfg["model_name"],
        "learning_rate": cfg["learning_rate"],
        "epochs": cfg["epochs"],
        "per_device_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "effective_batch_size": int(cfg["per_device_train_batch_size"]) * int(cfg["gradient_accumulation_steps"]),
        "lora_rank": cfg["lora_rank"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "max_length": cfg["max_length"],
        "precision": precision_name,
        "optimizer": cfg["optimizer"],
        "seed": cfg["seed"],
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(total_vram, 2),
        "trainable_params": trainable,
        "train_loss": train_result.metrics.get("train_loss"),
        "validation_loss": validation.get("eval_loss"),
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "max_memory_allocated_gb": round(max_alloc, 3),
        "adapter_path": str(latest_adapter.relative_to(ROOT)),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "training_log": str(training_log.relative_to(ROOT)),
    }
    append_result_csv(ROOT / "experiments" / "results.csv", result_row)

    print("\nTraining completed.")
    print(json.dumps(result_row, indent=2, ensure_ascii=False))
    print(f"Detailed run evidence: {run_dir}")
    print(f"Stable adapter path:   {latest_adapter}")

if __name__ == "__main__":
    main()
