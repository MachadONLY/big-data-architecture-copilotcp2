import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def load(name):
    return [
        json.loads(line)
        for line in (DATA / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_split_sizes_and_balance():
    train = load("train.jsonl")
    val = load("validation.jsonl")
    test = load("test.jsonl")
    assert len(train) == 280
    assert len(val) == 60
    assert len(test) == 60

    assert set(Counter(x["category"] for x in train).values()) == {14}
    assert set(Counter(x["category"] for x in val).values()) == {3}
    assert set(Counter(x["category"] for x in test).values()) == {3}


def test_no_id_leakage():
    train = {x["id"] for x in load("train.jsonl")}
    val = {x["id"] for x in load("validation.jsonl")}
    test = {x["id"] for x in load("test.jsonl")}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_required_fields():
    required = {
        "id", "category", "instruction", "input", "output",
        "expected_bottleneck_terms", "expected_pattern_terms",
        "must_include", "must_not_include", "prompt", "completion",
    }
    for filename in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        for row in load(filename):
            assert required <= row.keys()
            assert row["output"].strip()
            assert row["prompt"].strip()
            assert row["completion"].strip()
