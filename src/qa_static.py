"""Fast offline preflight: validates package sources, configs, dataset and documentation links.

Does NOT claim that any CUDA training, model download or Streamlit inference has run.
"""
from __future__ import annotations
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check() -> list[str]:
    issues = []
    for directory in ('src','app','tests'):
        for file in (ROOT/directory).rglob('*.py'):
            try:
                ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
            except SyntaxError as exc:
                issues.append(f'Invalid Python {file.relative_to(ROOT)}: {exc}')

    rows = {}
    for name, expected in [('train',280),('validation',60),('test',60)]:
        path = ROOT/'data'/'processed'/f'{name}.jsonl'
        try:
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            rows[name] = records
            if len(records) != expected:
                issues.append(f'{name} count {len(records)} not {expected}')
            if len({rec['id'] for rec in records}) != expected:
                issues.append(f'{name}: repeated IDs')
            counts = Counter(rec['category'] for rec in records)
            if len(counts) != 20 or len(set(counts.values())) != 1:
                issues.append(f'{name}: category imbalance')
            if any(rec['completion'] != rec['output'] or not rec['prompt'].endswith('### Answer\n') for rec in records):
                issues.append(f'{name}: prompt/completion mismatch')
        except (KeyError, ValueError, OSError) as exc:
            issues.append(f'{name}: {exc}')

    if len(rows) == 3:
        ids = [set(r['id'] for r in item) for item in rows.values()]
        if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
            issues.append('Dataset exact ID leakage')

    for file in (ROOT/'docs').glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#','mailto:')):
                continue
            path = (file.parent/target.split('#')[0]).resolve()
            if not path.exists():
                issues.append(f'Broken relative Markdown link in {file.name}: {target}')

    needed = [ROOT/'README.md',ROOT/'QUICKSTART.md',ROOT/'app/app.py',
              ROOT/'src/train.py',ROOT/'src/evaluate.py',ROOT/'src/gpu_check.py',
              ROOT/'scripts/run_pipeline.ps1',ROOT/'configs/experiment_01.yaml',
              ROOT/'configs/experiment_02.yaml',ROOT/'configs/experiment_03.yaml']
    for path in needed:
        if not path.is_file():
            issues.append(f'Missing project file {path.relative_to(ROOT)}')
    return issues

if __name__ == '__main__':
    err=check()
    if err:
        print('OFFLINE QA: FAIL')
        for e in err:
            print(' -', e)
        sys.exit(1)
    print('OFFLINE QA: PASS — syntax / dataset / structural documentation links')
    print('NOTE: Offline QA does NOT verify CUDA training or Streamlit inference.')
