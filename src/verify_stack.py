"""Verify the pinned Hugging Face/QLoRA stack before expensive training.

This is an API and kernel smoke test, not evidence that fine-tuning completed.
With --require-cuda it also runs a real NF4 quantize/dequantize operation on GPU.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import tempfile
from pathlib import Path

EXPECTED = {
    "transformers": "4.57.6",
    "trl": "0.26.2",
    "peft": "0.18.1",
    "datasets": "4.4.2",
    "accelerate": "1.12.0",
    "bitsandbytes": "0.48.2",
}


def collect(require_cuda: bool) -> dict:
    import torch
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig

    versions = {name: metadata.version(name) for name in ["torch", *EXPECTED]}
    mismatches = {
        name: {"expected": wanted, "actual": versions.get(name)}
        for name, wanted in EXPECTED.items()
        if versions.get(name) != wanted
    }

    api_checks: dict[str, object] = {}
    try:
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = SFTConfig(
                output_dir=temp_dir,
                max_length=1024,
                completion_only_loss=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                report_to="none",
                bf16=False,
                fp16=False,
            )
        api_checks["sft_config"] = bool(cfg.max_length == 1024 and cfg.completion_only_loss)
        api_checks["bitsandbytes_config"] = True
    except Exception as exc:
        api_checks["configuration_error"] = repr(exc)

    cuda = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        cuda["device"] = idx
        cuda["gpu_name"] = torch.cuda.get_device_name(idx)
        try:
            source = torch.randn(4096, device=f"cuda:{idx}", dtype=torch.float16)
            quantized, state = bnb.functional.quantize_4bit(source, quant_type="nf4")
            restored = bnb.functional.dequantize_4bit(quantized, state)
            torch.cuda.synchronize(idx)
            cuda["bitsandbytes_nf4_smoke"] = {
                "passed": bool(torch.isfinite(restored).all().item()),
                "mean_absolute_error": round(
                    float((restored.float() - source.float()).abs().mean().item()), 8
                ),
            }
            del source, quantized, restored
        except Exception as exc:
            cuda["bitsandbytes_nf4_smoke"] = {"passed": False, "error": repr(exc)}

    ready = bool(
        not mismatches
        and api_checks.get("sft_config")
        and api_checks.get("bitsandbytes_config")
        and (
            not require_cuda
            or (
                cuda["available"]
                and cuda.get("bitsandbytes_nf4_smoke", {}).get("passed")
            )
        )
    )
    return {
        "ready": ready,
        "require_cuda": require_cuda,
        "versions": versions,
        "expected_versions": EXPECTED,
        "version_mismatches": mismatches,
        "api_checks": api_checks,
        "cuda": cuda,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = collect(args.require_cuda)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
