"""Aggregate completed blind human-review CSVs.

Usage:
python src/score_manual_review.py --review reviews/reviewer_1.csv reviews/reviewer_2.csv --key experiments/evaluation/.../blind_key_DO_NOT_SHARE_WITH_REVIEWER.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

DIMENSIONS = [
    "bottleneck_0_4", "solution_0_4", "technical_justification_0_4",
    "relevance_0_4", "hallucination_absence_0_4", "completeness_0_4", "clarity_0_4",
]

def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--review", type=Path, nargs="+", required=True)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("experiments/manual_review_summary.json"))
    args = p.parse_args()

    key = {r["id"]: r for r in read_csv(args.key)}
    scores = {"base": defaultdict(list), "fine_tuned": defaultdict(list)}
    valid_rows = 0

    for review_file in args.review:
        for row in read_csv(review_file):
            if row["id"] not in key:
                continue
            mapping = key[row["id"]]
            for side in ("A", "B"):
                model = mapping[side]
                values = []
                complete = True
                for dim in DIMENSIONS:
                    raw = row.get(f"{side}_{dim}", "").strip()
                    if raw == "":
                        complete = False
                        break
                    value = float(raw)
                    if not 0 <= value <= 4:
                        raise ValueError(f"{review_file}: score outside 0-4 for {row['id']}")
                    values.append(value)
                if complete:
                    for dim, value in zip(DIMENSIONS, values):
                        scores[model][dim].append(value)
                    scores[model]["overall_mean"].append(statistics.mean(values))
            valid_rows += 1

    summary = {"review_files": [str(x) for x in args.review], "rows_seen": valid_rows}
    for model in ("base", "fine_tuned"):
        summary[model] = {
            k: round(statistics.mean(v), 4) if v else None
            for k, v in scores[model].items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
