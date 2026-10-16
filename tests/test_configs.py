from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_experiments_are_controlled():
    a = read("experiment_01.yaml")
    b = read("experiment_02.yaml")
    c = read("experiment_03.yaml")

    # A -> B: only LoRA capacity changes (plus experiment id)
    for key in a:
        if key not in {"experiment_id", "lora_rank", "lora_alpha"}:
            assert a[key] == b[key]

    # B -> C: only learning rate changes (plus experiment id)
    for key in b:
        if key not in {"experiment_id", "learning_rate"}:
            assert b[key] == c[key]

    assert a["lora_rank"] == 8
    assert b["lora_rank"] == c["lora_rank"] == 16
    assert b["learning_rate"] == 2e-4
    assert c["learning_rate"] == 1e-4
