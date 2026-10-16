"""Select an adapter using VALIDATION loss only (never the test set)."""
from __future__ import annotations
import csv

from common import ROOT

def main():
    results = ROOT / "experiments" / "results.csv"
    with results.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("validation_loss", "").strip()]
    if not rows:
        raise RuntimeError("No completed experiment with validation_loss found.")
    # Choose only the latest completed run for each experiment; avoid stale mixed-model records.
    latest = {}
    for row in rows:
        latest[row["experiment_id"]] = row
    expected = {"experiment_01", "experiment_02", "experiment_03"}
    if set(latest) != expected:
        raise RuntimeError(f"Expected completed experiments {sorted(expected)}, got {sorted(latest)}")
    if len({r["model"] for r in latest.values()}) != 1:
        raise RuntimeError("Cannot compare results from different base models. Re-run all three with the same model.")
    for row in latest.values():
        row["_val"] = float(row["validation_loss"])
    best = min(latest.values(), key=lambda r: r["_val"])
    adapter = ROOT / best["adapter_path"]
    if not adapter.exists():
        raise FileNotFoundError(f"Selected adapter path does not exist: {adapter}")
    pointer = ROOT / "experiments" / "best_adapter.txt"
    pointer.write_text(str(adapter.relative_to(ROOT)), encoding="utf-8")
    print(f"Selected by lowest validation loss: {best['experiment_id']}")
    print(f"validation_loss={best['_val']}")
    print(f"adapter={adapter}")

if __name__ == "__main__":
    main()
