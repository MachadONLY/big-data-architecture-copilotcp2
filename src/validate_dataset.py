"""Standalone dataset validation for presentation/demo."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

def load(name):
    return [json.loads(x) for x in (DATA / name).read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    splits = {n: load(f"{n}.jsonl") for n in ("train", "validation", "test")}
    ids = {k: {x["id"] for x in v} for k, v in splits.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
    for field in ("input", "prompt", "output"):
        values = [row[field] for records in splits.values() for row in records]
        assert len(values) == len(set(values)), f"exact {field} overlap detected"
    print("Dataset validation: PASS")
    for split, rows in splits.items():
        counts = Counter(x["category"] for x in rows)
        print(f"{split:10s}: {len(rows):3d} examples | {len(counts)} categories | per-category={sorted(set(counts.values()))}")
    print(f"TOTAL      : {sum(len(x) for x in splits.values())} examples")
    print("ID leakage : NONE")
    print("Exact text : NONE across input/prompt/output")

if __name__ == "__main__":
    main()
