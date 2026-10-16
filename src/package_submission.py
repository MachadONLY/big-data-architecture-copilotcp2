"""Create a lightweight submission ZIP without model weights or virtual environments.

Run only after the submission gate passes. This package is suitable as a Teams backup;
the GitHub repository remains the primary code/documentation deliverable.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from common import ROOT

DIST = ROOT / "dist"
OUT = DIST / "big-data-architecture-copilot-submission.zip"
STATUS = ROOT / "submission_status.json"

EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".ipynb_checkpoints", "checkpoints", "adapter", "cache", "outputs", "dist",
}
EXCLUDED_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".pyc"}


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def git_snapshot() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return f"commit={commit}\nworking_tree_clean={not bool(status)}\nstatus=\n{status}\n"
    except Exception:
        return "git repository/commit not available at packaging time\n"


def main() -> None:
    if not STATUS.exists():
        raise SystemExit("submission_status.json not found. Run scripts/submission_gate.ps1 first.")
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if not status.get("ready"):
        raise SystemExit("Submission gate has not passed. Refusing to package an incomplete submission.")

    DIST.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    # Produce hashes for the actual submission contents rather than shipping a
    # stale pre-training manifest. Do not alter the checked-in source tree.
    files = [p for p in ROOT.rglob("*") if include(p) and p.name != "MANIFEST.sha256"]
    manifest_lines: list[str] = []
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(files):
            arc = Path("big-data-architecture-copilot") / p.relative_to(ROOT)
            data = p.read_bytes()
            z.writestr(arc.as_posix(), data)
            manifest_lines.append(f"{hashlib.sha256(data).hexdigest()}  ./{p.relative_to(ROOT).as_posix()}")
        snapshot = git_snapshot().encode("utf-8")
        z.writestr("big-data-architecture-copilot/GIT_SNAPSHOT.txt", snapshot)
        manifest_lines.append(f"{hashlib.sha256(snapshot).hexdigest()}  ./GIT_SNAPSHOT.txt")
        z.writestr("big-data-architecture-copilot/MANIFEST.sha256", "\n".join(sorted(manifest_lines)) + "\n")

    print(f"Submission ZIP: {OUT}")
    print(f"Files included: {len(files) + 2}")
    print("Model weights/adapters/checkpoints are intentionally excluded from the code ZIP.")
    pointer = ROOT / "experiments" / "best_adapter.txt"
    if pointer.is_file():
        adapter_dir = ROOT / pointer.read_text(encoding="utf-8").strip()
        adapter_zip = DIST / "big-data-architecture-copilot-adapter.zip"
        if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").is_file():
            with zipfile.ZipFile(adapter_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as az:
                for f in sorted(adapter_dir.rglob("*")):
                    if f.is_file():
                        az.write(f, (Path("adapter") / f.relative_to(adapter_dir)).as_posix())
            print(f"Adapter for reproducible demo (separate artifact): {adapter_zip}")
            print("IMPORTANT: model base still needs to be downloaded separately.")


if __name__ == "__main__":
    main()
