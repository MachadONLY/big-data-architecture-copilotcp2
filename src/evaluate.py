"""Fixed test-set evaluation: BASE MODEL vs FINE-TUNED MODEL.

Automatic metrics are intentionally limited and transparent:
- bottleneck term hit
- architecture pattern term hit
- required concept coverage
- response-format compliance
- forbidden-claim hit
- ROUGE-L against the curated reference answer

They are NOT a replacement for the human rubric. The script also produces a
blind-review CSV so two reviewers can score both answers without knowing which
one is the fine-tuned model.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import ROOT, ensure_dir, write_json
from prompts import SECTION_NAMES, parse_sections

TEST_FILE = ROOT / "data" / "processed" / "test.jsonl"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def contains_any(text: str, terms: list[str]) -> float:
    n = normalize(text)
    return float(any(normalize(t) in n for t in terms))


def coverage(text: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    n = normalize(text)
    return sum(normalize(t) in n for t in terms) / len(terms)


def format_compliance(text: str) -> float:
    sections = parse_sections(text)
    present = sum(bool(sections[name].strip()) for name in SECTION_NAMES)
    return present / len(SECTION_NAMES)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_bundle(model_name: str, adapter_path: Path | None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official evaluation.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    source = str(adapter_path) if adapter_path else model_name
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map={"": 0},
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path)) if adapter_path else base
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_one(tokenizer, model, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    started = time.perf_counter()
    ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), elapsed


def run_predictions(model_name: str, adapter_path: Path | None, rows: list[dict], max_new_tokens: int) -> list[dict]:
    tokenizer, model = model_bundle(model_name, adapter_path)
    predictions = []
    for i, row in enumerate(rows, start=1):
        text, elapsed = generate_one(tokenizer, model, row["prompt"], max_new_tokens)
        predictions.append({"id": row["id"], "text": text, "elapsed_seconds": elapsed})
        print(f"[{i:02d}/{len(rows)}] {row['id']} - {elapsed:.2f}s")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def score_answer(row: dict, prediction: str, rouge) -> dict:
    bottleneck = contains_any(prediction, row["expected_bottleneck_terms"])
    pattern = contains_any(prediction, row["expected_pattern_terms"])
    concepts = coverage(prediction, row["must_include"])
    fmt = format_compliance(prediction)
    forbidden = contains_any(prediction, row["must_not_include"])
    rouge_l = rouge.score(row["output"], prediction)["rougeL"].fmeasure

    score = (
        0.25 * bottleneck
        + 0.25 * pattern
        + 0.20 * concepts
        + 0.15 * fmt
        + 0.15 * rouge_l
        - 0.10 * forbidden
    )
    score = max(0.0, min(1.0, score))
    return {
        "bottleneck_term_hit": bottleneck,
        "pattern_term_hit": pattern,
        "concept_coverage": concepts,
        "format_compliance": fmt,
        "forbidden_claim_hit": forbidden,
        "rougeL": rouge_l,
        "heuristic_auto_score": score,
    }


def mean(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(1, len(rows))


def write_blind_review(path: Path, key_path: Path, test_rows: list[dict], base_map: dict, fine_map: dict, seed: int) -> None:
    rng = random.Random(seed)
    score_fields = [
        "bottleneck_0_4", "solution_0_4", "technical_justification_0_4",
        "relevance_0_4", "hallucination_absence_0_4", "completeness_0_4", "clarity_0_4",
    ]
    fields = [
        "id", "category", "scenario", "reference_answer", "answer_A", "answer_B",
        "reviewer", *[f"A_{x}" for x in score_fields], *[f"B_{x}" for x in score_fields], "notes"
    ]
    key_rows = []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in test_rows:
            if rng.random() < 0.5:
                a, b = base_map[row["id"]]["text"], fine_map[row["id"]]["text"]
                mapping = ("base", "fine_tuned")
            else:
                a, b = fine_map[row["id"]]["text"], base_map[row["id"]]["text"]
                mapping = ("fine_tuned", "base")
            record = {
                "id": row["id"],
                "category": row["category"],
                "scenario": row["input"],
                "reference_answer": row["output"],
                "answer_A": a,
                "answer_B": b,
                "reviewer": "",
                "notes": "",
            }
            for field in fields:
                record.setdefault(field, "")
            w.writerow(record)
            key_rows.append({"id": row["id"], "A": mapping[0], "B": mapping[1]})

    with key_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "A", "B"])
        w.writeheader()
        w.writerows(key_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--model", default=None, help="Override only when it matches the adapter base model.")
    parser.add_argument("--max-new-tokens", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")
    config_path = args.adapter / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Trained adapter config not found: {config_path}")
    recorded_model = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    model_name = args.model or recorded_model
    if not model_name:
        raise ValueError("Cannot determine model name from adapter_config.json; pass --model.")
    if recorded_model and args.model and args.model != recorded_model:
        raise ValueError(f"Model mismatch: adapter expects {recorded_model}, received {args.model}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; evaluation aborted.")

    rows = load_jsonl(TEST_FILE)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ensure_dir(ROOT / "experiments" / "evaluation" / stamp)

    print("\n=== BASE MODEL ===")
    base_predictions = run_predictions(model_name, None, rows, args.max_new_tokens)
    print("\n=== FINE-TUNED MODEL ===")
    fine_predictions = run_predictions(model_name, args.adapter, rows, args.max_new_tokens)

    base_map = {x["id"]: x for x in base_predictions}
    fine_map = {x["id"]: x for x in fine_predictions}
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    comparison_rows = []
    base_scores = []
    fine_scores = []
    for row in rows:
        b = score_answer(row, base_map[row["id"]]["text"], rouge)
        f = score_answer(row, fine_map[row["id"]]["text"], rouge)
        base_scores.append(b)
        fine_scores.append(f)
        comparison_rows.append({
            "id": row["id"],
            "category": row["category"],
            "base_answer": base_map[row["id"]]["text"],
            "fine_tuned_answer": fine_map[row["id"]]["text"],
            "base_latency_s": base_map[row["id"]]["elapsed_seconds"],
            "fine_tuned_latency_s": fine_map[row["id"]]["elapsed_seconds"],
            **{f"base_{k}": v for k, v in b.items()},
            **{f"fine_{k}": v for k, v in f.items()},
            "auto_delta": f["heuristic_auto_score"] - b["heuristic_auto_score"],
        })

    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)

    with (out / "base_predictions.jsonl").open("w", encoding="utf-8") as f:
        for x in base_predictions:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with (out / "fine_tuned_predictions.jsonl").open("w", encoding="utf-8") as f:
        for x in fine_predictions:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    metrics = [
        "bottleneck_term_hit", "pattern_term_hit", "concept_coverage",
        "format_compliance", "forbidden_claim_hit", "rougeL", "heuristic_auto_score",
    ]
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "adapter": str(args.adapter),
        "test_examples": len(rows),
        "decoding": {
            "do_sample": False,
            "repetition_penalty": 1.05,
            "max_new_tokens": args.max_new_tokens,
        },
        "base": {m: mean(base_scores, m) for m in metrics},
        "fine_tuned": {m: mean(fine_scores, m) for m in metrics},
        "important_limitations": [
            "Keyword hits do not prove semantic correctness.",
            "ROUGE-L measures surface overlap, not architectural truth.",
            "The heuristic score is a transparent project metric, not an industry benchmark.",
            "Human blind scoring is required for the final academic conclusion.",
        ],
    }
    summary["delta_fine_minus_base"] = {
        m: summary["fine_tuned"][m] - summary["base"][m] for m in metrics
    }
    write_json(out / "summary.json", summary)

    worsened = [r for r in comparison_rows if r["auto_delta"] < 0]
    with (out / "worsened_cases.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(worsened)

    write_blind_review(
        out / "blind_review.csv",
        out / "blind_key_DO_NOT_SHARE_WITH_REVIEWER.csv",
        rows, base_map, fine_map, args.seed,
    )

    print("\nEvaluation completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nAll evaluation artifacts: {out}")
    print("Fill blind_review.csv with human 0-4 scores, then run score_manual_review.py.")

if __name__ == "__main__":
    main()
