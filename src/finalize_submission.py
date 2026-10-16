"""Generate a measured final report after real GPU experiments.

This script never fabricates numbers. It only summarizes artifacts that already exist.
It can be run repeatedly after the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT

RESULTS = ROOT / "experiments" / "results.csv"
GPU_JSON = ROOT / "assets" / "screenshots" / "gpu_environment.json"
STACK_JSON = ROOT / "assets" / "screenshots" / "stack_environment.json"
BEST_ADAPTER = ROOT / "experiments" / "best_adapter.txt"
FINAL_REPORT = ROOT / "FINAL_REPORT.md"
STATUS_JSON = ROOT / "submission_status.json"
README = ROOT / "README.md"
SCREENSHOT = ROOT / "assets" / "screenshots" / "gpu_training_vscode.png"
FRONTEND_SCREENSHOT = ROOT / "assets" / "screenshots" / "frontend_demo.png"


def read_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    with RESULTS.open("r", encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("experiment_id")]


def latest_evaluation() -> tuple[Path | None, dict | None]:
    root = ROOT / "experiments" / "evaluation"
    if not root.exists():
        return None, None
    candidates = sorted(root.glob("*/summary.json"), reverse=True)
    if not candidates:
        return None, None
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def fmt(v, digits=4):
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)




def valid_number(value, *, positive: bool = False) -> bool:
    try:
        n = float(value)
        return math.isfinite(n) and (n > 0 if positive else n >= 0)
    except (TypeError, ValueError):
        return False


def valid_experiment_rows(rows: list[dict]) -> bool:
    expected = {"experiment_01", "experiment_02", "experiment_03"}
    present = set()
    for row in rows:
        if row.get("experiment_id") not in expected:
            continue
        if not all(valid_number(row.get(k), positive=True) for k in (
            "train_runtime_seconds", "max_memory_allocated_gb", "vram_gb",
        )):
            continue
        if not all(valid_number(row.get(k)) for k in ("train_loss", "validation_loss")):
            continue
        if not row.get("gpu", "").strip():
            continue
        present.add(row["experiment_id"])
    return present == expected


def adapter_is_real(pointer: str | None) -> bool:
    if not pointer:
        return False
    candidate = ROOT / pointer
    if not candidate.is_dir() or not (candidate / "adapter_config.json").is_file():
        return False
    return any((candidate / filename).is_file() and (candidate / filename).stat().st_size > 0
               for filename in ("adapter_model.safetensors", "adapter_model.bin"))


def valid_evaluation(path: Path | None, data: dict | None) -> bool:
    if not path or not data or data.get("test_examples") != 60:
        return False
    if not all(isinstance(data.get(k), dict) for k in ("base", "fine_tuned", "delta_fine_minus_base")):
        return False
    if not all(valid_number(data[k].get("heuristic_auto_score")) for k in ("base", "fine_tuned")):
        return False
    comparison = path.parent / "comparison.csv"
    if not comparison.is_file():
        return False
    with comparison.open("r", encoding="utf-8", newline="") as f:
        return len(list(csv.DictReader(f))) == 60


def valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8000:
        return False
    with path.open("rb") as f:
        header = f.read(24)
    if not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR":
        return False
    width, height = struct.unpack(">II", header[16:24])
    return width >= 800 and height >= 450


def valid_members(members: list[str]) -> bool:
    if not 4 <= len(members) <= 5:
        return False
    return all(len(m.strip()) > 8 and re.search(r"\bRM\s*\d{4,}\b", m, re.IGNORECASE)
               and not re.search(r"Nome\s+\d|XXXXX|\.\.\.", m, re.IGNORECASE)
               for m in members)


def valid_stack_evidence(data: dict | None) -> bool:
    return bool(
        data
        and data.get("ready") is True
        and data.get("require_cuda") is True
        and data.get("cuda", {}).get("available") is True
        and data.get("cuda", {}).get("bitsandbytes_nf4_smoke", {}).get("passed") is True
        and not data.get("version_mismatches")
    )


def valid_run_evidence(rows: list[dict]) -> bool:
    latest = {}
    for row in rows:
        if row.get("experiment_id") in {"experiment_01", "experiment_02", "experiment_03"}:
            latest[row["experiment_id"]] = row
    if set(latest) != {"experiment_01", "experiment_02", "experiment_03"}:
        return False
    for experiment_id, row in latest.items():
        run_dir_text = row.get("run_dir", "")
        log_text = row.get("training_log", "")
        if not run_dir_text or not log_text:
            return False
        run_dir = ROOT / run_dir_text
        log_file = ROOT / log_text
        if not run_dir.is_dir() or not log_file.is_file() or log_file.stat().st_size < 200:
            return False
        for name in ("config_used.yaml", "environment.json", "metrics.json"):
            if not (run_dir / name).is_file():
                return False
        try:
            environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not environment.get("gpu", {}).get("cuda_available"):
            return False
        if metrics.get("experiment_id") != experiment_id:
            return False
    return True


def git_evidence() -> dict:
    def command(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=30
        ).strip()

    try:
        head = command("rev-parse", "HEAD")
        branch = command("branch", "--show-current")
        remote = command("remote", "get-url", "origin")
        status = command("status", "--porcelain")
        remote_line = command("ls-remote", "origin", f"refs/heads/{branch}")
        remote_head = remote_line.split()[0] if remote_line else ""
        return {
            "head": head,
            "branch": branch,
            "origin": remote,
            "working_tree_clean": not bool(status),
            "remote_head": remote_head,
            "pushed": bool(head == remote_head and branch and re.search(r"github\.com[:/]", remote, re.I)),
        }
    except Exception as exc:
        return {"pushed": False, "error": repr(exc)}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--member", action="append", default=[], help="Repeat for each member, e.g. 'Gabriel Machado — RM12345'.")
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()

    # Read project roster by default, avoiding duplicate/manual data entry on Windows.
    # Explicit --member arguments remain available for groups using another roster.
    if not args.member:
        roster_path = ROOT / "team_members.json"
        if roster_path.is_file():
            raw_members = json.loads(roster_path.read_text(encoding="utf-8"))
            args.member = [
                f"{str(item['name']).strip()} — RM{str(item['rm']).strip().removeprefix('RM')}"
                for item in raw_members
            ]

    results = read_results()
    eval_path, evaluation = latest_evaluation()
    gpu = json.loads(GPU_JSON.read_text(encoding="utf-8")) if GPU_JSON.exists() else None
    stack = json.loads(STACK_JSON.read_text(encoding="utf-8")) if STACK_JSON.exists() else None
    best_adapter = BEST_ADAPTER.read_text(encoding="utf-8").strip() if BEST_ADAPTER.exists() else None
    git = git_evidence()

    checks = {
        "three_experiments": valid_experiment_rows(results),
        "training_run_evidence": valid_run_evidence(results),
        "gpu_json": bool(gpu and gpu.get("cuda_available") and gpu.get("gpu_name") and gpu.get("nvidia_smi", {}).get("returncode") == 0),
        "qlora_stack_nf4": valid_stack_evidence(stack),
        "best_adapter": adapter_is_real(best_adapter),
        "evaluation_summary": valid_evaluation(eval_path, evaluation),
        "gpu_training_screenshot": valid_png(SCREENSHOT),
        "frontend_screenshot": valid_png(FRONTEND_SCREENSHOT),
        "environment_lock": (ROOT / "environment.lock.txt").is_file(),
        "team_members": valid_members(args.member),
        "github_pushed": bool(git.get("pushed") and git.get("working_tree_clean")),
    }
    ready = all(checks.values())

    lines = [
        "# FINAL REPORT — Big Data Architecture Copilot",
        "",
        f"> Generated from measured project artifacts at {datetime.now(timezone.utc).isoformat()}.",
        "> No metric in this report is invented by this script.",
        "",
        "## Submission status",
        "",
        f"**{'READY' if ready else 'NOT READY'}**",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for key, ok in checks.items():
        lines.append(f"| {key} | {'PASS' if ok else 'MISSING'} |")

    lines += ["", "## Team", ""]
    if args.member:
        lines.extend([f"- {m}" for m in args.member])
    else:
        lines.append("- **MISSING: provide member names and RM values when finalizing.**")

    lines += ["", "## GPU evidence", ""]
    if gpu:
        lines += [
            f"- CUDA available: `{gpu.get('cuda_available')}`",
            f"- GPU: `{gpu.get('gpu_name', '—')}`",
            f"- VRAM: `{gpu.get('total_vram_gb', '—')} GB`",
            f"- PyTorch: `{gpu.get('torch_version', '—')}`",
            f"- torch CUDA: `{gpu.get('torch_cuda_version', '—')}`",
            f"- BF16 supported: `{gpu.get('bf16_supported', '—')}`",
        ]
    else:
        lines.append("**MISSING: assets/screenshots/gpu_environment.json**")

    lines += ["", "## Reproducibility and publication", ""]
    lines += [
        f"- Pinned QLoRA/NF4 stack: `{'PASS' if valid_stack_evidence(stack) else 'MISSING'}`",
        f"- Timestamped training logs: `{'PASS' if valid_run_evidence(results) else 'MISSING'}`",
        f"- GitHub origin: `{git.get('origin', 'MISSING')}`",
        f"- Commit pushed and working tree clean: `{bool(git.get('pushed') and git.get('working_tree_clean'))}`",
    ]

    lines += ["", "## Experiments", ""]
    if results:
        lines += [
            "| Exp | Model | LR | Epochs | Rank | Alpha | Precision | Train loss | Val loss | Runtime (s) | Peak VRAM (GB) |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for r in results:
            lines.append(
                "| {experiment_id} | {model} | {learning_rate} | {epochs} | {lora_rank} | {lora_alpha} | {precision} | {train_loss} | {validation_loss} | {runtime} | {peak} |".format(
                    experiment_id=r.get("experiment_id", "—"),
                    model=r.get("model", "—"),
                    learning_rate=r.get("learning_rate", "—"),
                    epochs=r.get("epochs", "—"),
                    lora_rank=r.get("lora_rank", "—"),
                    lora_alpha=r.get("lora_alpha", "—"),
                    precision=r.get("precision", "—"),
                    train_loss=fmt(r.get("train_loss")),
                    validation_loss=fmt(r.get("validation_loss")),
                    runtime=fmt(r.get("train_runtime_seconds"), 2),
                    peak=fmt(r.get("max_memory_allocated_gb"), 3),
                )
            )
    else:
        lines.append("**MISSING: no measured experiment rows in experiments/results.csv.**")

    lines += ["", "## Selected adapter", ""]
    lines.append(f"`{best_adapter}`" if best_adapter else "**MISSING: experiments/best_adapter.txt**")

    lines += ["", "## Base vs Fine-tuned evaluation", ""]
    if evaluation:
        lines.append(f"Source: `{eval_path.relative_to(ROOT)}`")
        lines.append("")
        lines += [
            "| Metric | Base | Fine-tuned | Delta |",
            "|---|---:|---:|---:|",
        ]
        base = evaluation.get("base", {})
        fine = evaluation.get("fine_tuned", {})
        delta = evaluation.get("delta_fine_minus_base", {})
        for metric in sorted(set(base) | set(fine) | set(delta)):
            lines.append(f"| {metric} | {fmt(base.get(metric))} | {fmt(fine.get(metric))} | {fmt(delta.get(metric))} |")
        counts = evaluation.get("qualitative_case_counts", {})
        if counts:
            lines += [
                "",
                "Automatic case-level comparison (used to select qualitative examples):",
                "",
                f"- Improved: `{counts.get('improved', '—')}`",
                f"- Tied: `{counts.get('tied', '—')}`",
                f"- Worsened: `{counts.get('worsened', '—')}`",
            ]
    else:
        lines.append("**MISSING: no evaluation summary found.**")

    lines += [
        "",
        "## Evidence files",
        "",
        "- `experiments/results.csv`",
        "- `experiments/best_adapter.txt`",
        "- `experiments/evaluation/<run>/summary.json`",
        "- `experiments/evaluation/<run>/comparison.csv`",
        "- `experiments/evaluation/<run>/worsened_cases.csv`",
        "- `assets/screenshots/gpu_environment.json`",
        "- `assets/screenshots/stack_environment.json`",
        "- `assets/screenshots/gpu_training_vscode.png`",
        "- `assets/screenshots/frontend_demo.png`",
        "- `environment.lock.txt`",
        "",
        "## Academic limitations",
        "",
        "Automatic keyword and ROUGE metrics are supporting evidence, not proof of architectural truth. The final interpretation must acknowledge failure cases and the synthetic nature of the dataset.",
        "",
    ]

    FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    STATUS_JSON.write_text(json.dumps({"ready": ready, "checks": checks}, indent=2), encoding="utf-8")

    # Update only the clearly marked status table; never insert measured values into README unless artifacts exist.
    if README.exists():
        text = README.read_text(encoding="utf-8")
        if valid_experiment_rows(results):
            text = text.replace("| Training loss real | **Pendente de execução em GPU** |", "| Training loss real | **Executado — ver experiments/results.csv** |")
            text = text.replace("| Validation loss real | **Pendente de execução em GPU** |", "| Validation loss real | **Executado — ver experiments/results.csv** |")
            text = text.replace("| Duração/VRAM real | **Pendente de execução em GPU** |", "| Duração/VRAM real | **Executado — ver experiments/results.csv** |")
        if valid_evaluation(eval_path, evaluation):
            text = text.replace("| Benchmark real | **Pendente de adapter treinado** |", "| Benchmark real | **Executado — ver experiments/evaluation/** |")
        if valid_png(SCREENSHOT):
            text = text.replace("| Screenshots CUDA | **Pendentes de execução na máquina** |", "| Screenshots CUDA | **Adicionados em assets/screenshots/** |")
        README.write_text(text, encoding="utf-8")

    print(FINAL_REPORT)
    print(json.dumps({"ready": ready, "checks": checks}, indent=2))
    if not ready and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
