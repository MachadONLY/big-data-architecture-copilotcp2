"""Local inference for the fine-tuned Big Data Architecture Copilot."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import ROOT
from prompts import build_prompt, parse_sections


class LocalArchitectureCopilot:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B-Base",
        adapter_path: str | Path | None = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. The submitted demo is configured for local GPU inference."
            )
        self.model_name = model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device_index = 0
        self.gpu_name = torch.cuda.get_device_name(self.device_index)
        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.precision = "bf16" if self.compute_dtype == torch.bfloat16 else "fp16"

        tokenizer_source = str(self.adapter_path) if self.adapter_path and self.adapter_path.exists() else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.compute_dtype,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": 0},
            dtype=self.compute_dtype,
            low_cpu_mem_usage=True,
        )
        if self.adapter_path:
            if not self.adapter_path.exists():
                raise FileNotFoundError(f"Adapter not found: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
        else:
            self.model = base
        self.model.eval()

    @torch.inference_mode()
    def analyze(self, scenario: str, max_new_tokens: int = 450) -> dict:
        instruction = (
            "Analise o cenário de arquitetura de dados abaixo e proponha uma solução "
            "tecnicamente justificável."
        )
        prompt = build_prompt(instruction, scenario)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        torch.cuda.synchronize()
        started = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "text": text,
            "sections": parse_sections(text),
            "elapsed_seconds": elapsed,
            "gpu": self.gpu_name,
            "model": self.model_name,
            "adapter": str(self.adapter_path) if self.adapter_path else None,
            "precision": self.precision,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--adapter",
        default=str(ROOT / "experiments" / "experiment_02" / "adapter"),
    )
    args = parser.parse_args()

    copilot = LocalArchitectureCopilot(args.model, args.adapter)
    result = copilot.analyze(args.scenario)
    print(result["text"])
    print(f"\n[{result['elapsed_seconds']:.2f}s | {result['gpu']} | local CUDA]")

if __name__ == "__main__":
    main()
