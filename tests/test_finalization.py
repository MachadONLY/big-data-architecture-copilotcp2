"""Submission gate must reject empty or misleading artifacts (CPU-only QA)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import finalize_submission as f


def exp_row(name: str) -> dict:
    return {
        "experiment_id": name, "train_loss": "1.2", "validation_loss": "1.1",
        "train_runtime_seconds": "12", "max_memory_allocated_gb": "2.2",
        "vram_gb": "8", "gpu": "RTX Test",
    }


def test_three_distinct_experiments_not_three_duplicates():
    same = exp_row("experiment_01")
    assert not f.valid_experiment_rows([same, same, same])
    assert f.valid_experiment_rows([exp_row(x) for x in (
        "experiment_01", "experiment_02", "experiment_03",
    )])


def test_nan_and_zero_runtime_fail():
    rows = [exp_row(x) for x in ("experiment_01", "experiment_02", "experiment_03")]
    rows[0]["validation_loss"] = "nan"
    assert not f.valid_experiment_rows(rows)
    rows[0]["validation_loss"] = "1.1"
    rows[1]["train_runtime_seconds"] = "0"
    assert not f.valid_experiment_rows(rows)


def test_member_guard_rejects_placeholders_and_short_groups():
    assert not f.valid_members(["Nome 1 — RM...", "Nome 2 — RM...", "Nome 3 — RM...", "Nome 4 — RM..."])
    assert not f.valid_members(["Gabriel Machado — RM12345"])
    assert f.valid_members([
        "Gabriel Machado — RM12345", "Ana Moraes — RM22222",
        "Joao Silva — RM33333", "Maria Costa — RM44444",
    ])


def test_adapter_guard_checks_weight_files(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "ROOT", tmp_path)
    d = tmp_path / "experiments" / "experiment_01" / "adapter"
    d.mkdir(parents=True)
    assert not f.adapter_is_real("experiments/experiment_01/adapter")
    (d / "adapter_config.json").write_text('{}')
    assert not f.adapter_is_real("experiments/experiment_01/adapter")
    (d / "adapter_model.safetensors").write_bytes(b"adapter-placeholder")
    assert f.adapter_is_real("experiments/experiment_01/adapter")


def test_evaluation_must_be_full_60_rows(tmp_path):
    loc = tmp_path / 'summary.json'
    loc.write_text('{}')
    summ = {'test_examples': 60, 'base': {'heuristic_auto_score': 0.1},
            'fine_tuned': {'heuristic_auto_score': 0.2}, 'delta_fine_minus_base': {}}
    assert not f.valid_evaluation(loc, summ)
    with (tmp_path / "comparison.csv").open('w', newline='') as h:
        w = csv.writer(h)
        w.writerow(['id'])
        w.writerows([[i] for i in range(60)])
    assert f.valid_evaluation(loc, summ)
    assert not f.valid_evaluation(loc, {**summ, 'test_examples': 10})


def test_screenshot_cannot_be_just_empty_filename(tmp_path):
    p = tmp_path / "gpu_training_vscode.png"
    p.write_bytes(b'')
    assert not f.valid_png(p)
    p.write_bytes(b"not a PNG"*3000)
    assert not f.valid_png(p)


def test_stack_evidence_requires_real_cuda_nf4_smoke():
    good = {
        "ready": True,
        "require_cuda": True,
        "version_mismatches": {},
        "cuda": {"available": True, "bitsandbytes_nf4_smoke": {"passed": True}},
    }
    assert f.valid_stack_evidence(good)
    assert not f.valid_stack_evidence({**good, "require_cuda": False})
    assert not f.valid_stack_evidence({**good, "version_mismatches": {"trl": {}}})


def test_run_evidence_requires_log_environment_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "ROOT", tmp_path)
    rows = []
    for name in ("experiment_01", "experiment_02", "experiment_03"):
        run = tmp_path / "experiments" / name / "runs" / "20261015T220000Z"
        run.mkdir(parents=True)
        (run / "training.log").write_text("measured training output\n" * 20)
        (run / "config_used.yaml").write_text("seed: 42\n")
        (run / "environment.json").write_text(json.dumps({"gpu": {"cuda_available": True}}))
        (run / "metrics.json").write_text(json.dumps({"experiment_id": name}))
        rows.append({
            "experiment_id": name,
            "run_dir": str(run.relative_to(tmp_path)),
            "training_log": str((run / "training.log").relative_to(tmp_path)),
        })
    assert f.valid_run_evidence(rows)
    (tmp_path / rows[0]["training_log"]).unlink()
    assert not f.valid_run_evidence(rows)


def test_clean_unexecuted_repo_is_not_ready():
    assert not f.valid_experiment_rows(f.read_results())
    assert not f.adapter_is_real(None)


def test_project_contains_train_validation_test_in_separate_files():
    root = Path(__file__).resolve().parents[1]
    counts = {}
    ids = {}
    for split, size in (("train", 280), ("validation", 60), ("test", 60)):
        data = root / "data" / "processed" / f"{split}.jsonl"
        records = [json.loads(row) for row in data.read_text(encoding='utf-8').splitlines()]
        assert len(records) == size
        assert all(x['prompt'].endswith('### Answer\n') for x in records)
        assert all(x['completion'] == x['output'] for x in records)
        counts[split] = len(records)
        ids[split] = {x['id'] for x in records}
    assert set.intersection(*ids.values()) == set()


def test_roster_has_no_fabricated_members():
    roster = json.loads((Path(__file__).resolve().parents[1] / "team_members.json").read_text(encoding="utf-8"))
    assert 3 <= len(roster) <= 5
    assert all(set(member) == {"name", "rm"} and str(member["name"]).strip() and str(member["rm"]).isdigit() for member in roster)
    assert roster[0] == {"name": "Camila do Prado Padalino", "rm": "98316"}
    assert roster[1] == {"name": "Gabriel Teixeira Machado", "rm": "551570"}
    assert roster[2] == {"name": "Felipe Cavalcante Bressane", "rm": "97688"}
