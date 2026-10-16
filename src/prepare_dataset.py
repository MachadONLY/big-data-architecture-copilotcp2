"""Create the project's deterministic, stratified JSONL dataset.

The source matrix lives in data/raw/scenario_blueprints.json.
This script creates exactly 400 examples: 20 categories x 20 contextualized scenarios.
Each category is split 14 train / 3 validation / 3 test with a fixed seed.

IMPORTANT:
- test.jsonl is never loaded by train.py.
- the dataset is synthetic and project-authored; the team should review it before submission.
- generation is deterministic so the professor can reproduce the exact split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from prompts import build_prompt

ROOT = Path(__file__).resolve().parents[1]
BLUEPRINTS = ROOT / "data" / "raw" / "scenario_blueprints.json"
OUT_DIR = ROOT / "data" / "processed"

INSTRUCTIONS = [
    "Analise o cenário de arquitetura de dados abaixo e proponha uma solução tecnicamente justificável.",
    "Diagnostique o problema arquitetural descrito e responda no formato solicitado.",
    "Atue como arquiteto de dados: identifique o gargalo, a causa raiz e uma solução com trade-offs.",
    "Avalie o cenário abaixo considerando escalabilidade, confiabilidade e operação.",
]

OPENERS = [
    "Neste cenário, o ponto central é",
    "A evidência principal indica",
    "O diagnóstico mais provável é",
    "Arquiteturalmente, o problema dominante é",
]

def join_items(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " e " + items[-1]

def reference_answer(spec: dict, ctx: dict, variant: int) -> str:
    techs = join_items(spec["techs"])
    advantages = join_items(spec["advantages"])
    tradeoffs = join_items(spec["tradeoffs"])
    contextual = (
        f"Considerando {ctx['scale']} no contexto de {ctx['industry']}, "
        f"a solução precisa respeitar {ctx['constraint']}."
    )
    if variant % 4 == 0:
        justification = f"{OPENERS[0]} {spec['root']} {contextual}"
    elif variant % 4 == 1:
        justification = f"{OPENERS[1]} {spec['root']} {contextual}"
    elif variant % 4 == 2:
        justification = f"{OPENERS[2]} {spec['root']} {contextual}"
    else:
        justification = f"{OPENERS[3]} {spec['root']} {contextual}"
    return (
        f"Gargalo: {spec['bottleneck']}\n"
        f"Causa raiz: {spec['root']}\n"
        f"Padrão recomendado: {spec['pattern']}\n"
        f"Solução: {spec['solution']} {contextual}\n"
        f"Justificativa técnica: {justification}\n"
        f"Tecnologias possíveis: {techs}.\n"
        f"Vantagens: {advantages}.\n"
        f"Limitações/Trade-offs: {tradeoffs}."
    )

def build_examples(blueprints: dict) -> list[dict]:
    records: list[dict] = []
    contexts = blueprints["contexts"]
    for category, spec in blueprints["categories"].items():
        for i, ctx in enumerate(contexts):
            stem = spec["stems"][i % len(spec["stems"])]
            instruction = INSTRUCTIONS[i % len(INSTRUCTIONS)]
            scenario = (
                f"Empresa do setor de {ctx['industry']}. "
                f"Escala atual: {ctx['scale']}. "
                f"{stem} Restrição relevante: {ctx['constraint']}."
            )
            output = reference_answer(spec, ctx, i)
            record_id = f"{category}-{i+1:03d}"
            difficulty = ["easy", "medium", "medium", "hard"][i % 4]
            records.append({
                "id": record_id,
                "category": category,
                "category_title": spec["title"],
                "difficulty": difficulty,
                "source_type": "synthetic_project_authored",
                "instruction": instruction,
                "input": scenario,
                "output": output,
                "expected_bottleneck_terms": spec["b_terms"],
                "expected_pattern_terms": spec["p_terms"],
                "must_include": spec["must"],
                "must_not_include": spec["forbidden"],
                "prompt": build_prompt(instruction, scenario),
                "completion": output,
            })
    return records

def stratified_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {}
    for rec in records:
        by_category.setdefault(rec["category"], []).append(rec)

    splits = {"train": [], "validation": [], "test": []}
    for category in sorted(by_category):
        items = list(by_category[category])
        rng.shuffle(items)
        assert len(items) == 20, (category, len(items))
        splits["train"].extend(items[:14])
        splits["validation"].extend(items[14:17])
        splits["test"].extend(items[17:20])

    for name in splits:
        rng.shuffle(splits[name])
    return splits

def validate(splits: dict[str, list[dict]]) -> None:
    required = {
        "id", "category", "difficulty", "instruction", "input", "output",
        "expected_bottleneck_terms", "expected_pattern_terms",
        "must_include", "must_not_include", "prompt", "completion",
    }
    ids: dict[str, set[str]] = {}
    for split_name, rows in splits.items():
        ids[split_name] = set()
        for row in rows:
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{row.get('id')} missing fields: {sorted(missing)}")
            if row["id"] in ids[split_name]:
                raise ValueError(f"duplicate id: {row['id']}")
            ids[split_name].add(row["id"])
            if not row["output"].strip():
                raise ValueError(f"empty output: {row['id']}")
        counts = Counter(r["category"] for r in rows)
        expected = 14 if split_name == "train" else 3
        if set(counts.values()) != {expected}:
            raise ValueError(f"split {split_name} is not balanced: {counts}")

    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("data leakage: the same id exists in more than one split")

    all_rows = [row for rows in splits.values() for row in rows]
    for field in ("input", "prompt", "output"):
        values = [row[field] for row in all_rows]
        if len(set(values)) != len(values):
            raise ValueError(f"exact duplicate detected in field: {field}")

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_qa_report(splits: dict[str, list[dict]], seed: int) -> None:
    all_rows = [row for rows in splits.values() for row in rows]
    report = {
        "seed": seed,
        "total_examples": sum(len(v) for v in splits.values()),
        "unique_inputs": len({row["input"] for row in all_rows}),
        "unique_prompts": len({row["prompt"] for row in all_rows}),
        "unique_outputs": len({row["output"] for row in all_rows}),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_sha256": {
            split: sha256_file(OUT_DIR / f"{split}.jsonl") for split in splits
        },
        "category_counts": {
            split: dict(sorted(Counter(r["category"] for r in rows).items()))
            for split, rows in splits.items()
        },
        "leakage_check": "PASS",
        "exact_text_overlap_check": "PASS",
        "required_fields_check": "PASS",
        "note": (
            "Dataset sintético e versionado. A qualidade semântica deve ser revisada "
            "pela equipe antes da submissão acadêmica."
        ),
    }
    (OUT_DIR / "dataset_qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    review_path = OUT_DIR / "manual_review_checklist.csv"
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "split", "category", "reviewer", "technical_correctness_0_4", "notes"],
        )
        writer.writeheader()
        for split, rows in splits.items():
            for row in rows:
                writer.writerow({
                    "id": row["id"],
                    "split": split,
                    "category": row["category"],
                    "reviewer": "",
                    "technical_correctness_0_4": "",
                    "notes": "",
                })

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    blueprints = json.loads(BLUEPRINTS.read_text(encoding="utf-8"))
    records = build_examples(blueprints)
    splits = stratified_split(records, args.seed)
    validate(splits)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(OUT_DIR / f"{split}.jsonl", rows)
    write_qa_report(splits, args.seed)

    print("Dataset generated successfully.")
    print(f"train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}")
    print(f"total={sum(len(v) for v in splits.values())}")
    print("No ID overlap between train/validation/test.")

if __name__ == "__main__":
    main()
