"""Generate presentation-ready charts from REAL experiment/evaluation outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT, ensure_dir


def plot_experiments(results_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_csv)
    if df.empty or df["train_loss"].isna().all():
        raise RuntimeError("experiments/results.csv has no real training results yet.")
    # Re-runs append rows. Presentation charts compare only the latest completed
    # run for each controlled experiment.
    df = (
        df.dropna(subset=["experiment_id", "train_loss", "validation_loss"])
        .drop_duplicates(subset=["experiment_id"], keep="last")
        .sort_values("experiment_id")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df["train_loss"], width, label="Training loss")
    ax.bar([i + width/2 for i in x], df["validation_loss"], width, label="Validation loss")
    ax.set_xticks(list(x), df["experiment_id"])
    ax.set_ylabel("Loss")
    ax.set_title("Training vs validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "experiment_losses.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["experiment_id"], df["train_runtime_seconds"])
    ax.set_ylabel("Seconds")
    ax.set_title("Training duration by experiment")
    fig.tight_layout()
    fig.savefig(out_dir / "training_duration.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["experiment_id"], df["max_memory_allocated_gb"])
    ax.set_ylabel("GB")
    ax.set_title("Peak allocated VRAM")
    fig.tight_layout()
    fig.savefig(out_dir / "peak_vram.png", dpi=180)
    plt.close(fig)


def plot_evaluation(summary_path: Path, out_dir: Path) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = [
        "bottleneck_term_hit", "pattern_term_hit", "concept_coverage",
        "format_compliance", "rougeL", "heuristic_auto_score",
    ]
    frame = pd.DataFrame({
        "metric": metrics,
        "Base": [data["base"][m] for m in metrics],
        "Fine-tuned": [data["fine_tuned"][m] for m in metrics],
    })
    ax = frame.set_index("metric").plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Base vs Fine-tuned — automatic benchmark")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "base_vs_finetuned.png", dpi=180)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-summary", type=Path, default=None)
    args = p.parse_args()
    out = ensure_dir(ROOT / "assets" / "charts")
    plot_experiments(ROOT / "experiments" / "results.csv", out)
    if args.evaluation_summary:
        plot_evaluation(args.evaluation_summary, out)
    print(f"Charts saved to {out}")

if __name__ == "__main__":
    main()
