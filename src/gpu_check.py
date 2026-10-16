"""Print and optionally save evidence that PyTorch can use the CUDA GPU."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

try:
    import torch
except Exception as exc:
    print("ERROR: PyTorch is not importable.")
    print(repr(exc))
    raise SystemExit(3)

def run_cmd(args: list[str]) -> dict:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {
            "command": " ".join(args),
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except FileNotFoundError:
        return {"command": " ".join(args), "returncode": None, "stdout": "", "stderr": "command not found"}
    except Exception as exc:
        return {"command": " ".join(args), "returncode": None, "stdout": "", "stderr": repr(exc)}

def collect() -> dict:
    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        data.update({
            "current_device": idx,
            "gpu_name": torch.cuda.get_device_name(idx),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_gb": round(props.total_memory / (1024**3), 2),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "allocated_vram_gb": round(torch.cuda.memory_allocated(idx) / (1024**3), 4),
            "reserved_vram_gb": round(torch.cuda.memory_reserved(idx) / (1024**3), 4),
            "cudnn_version": torch.backends.cudnn.version(),
        })
        try:
            torch.manual_seed(42)
            started = time.perf_counter()
            a = torch.randn((512, 512), device=f"cuda:{idx}", dtype=torch.float16)
            b = torch.randn((512, 512), device=f"cuda:{idx}", dtype=torch.float16)
            result = a @ b
            torch.cuda.synchronize(idx)
            data["cuda_compute_smoke"] = {
                "passed": bool(torch.isfinite(result).all().item()),
                "operation": "512x512 FP16 matrix multiplication",
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "checksum": round(float(result.float().mean().item()), 6),
            }
            del a, b, result
        except Exception as exc:
            data["cuda_compute_smoke"] = {"passed": False, "error": repr(exc)}
    data["nvidia_smi"] = run_cmd(["nvidia-smi"])
    data["nvcc_version"] = run_cmd(["nvcc", "--version"])
    return data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON evidence path")
    parser.add_argument("--require-cuda", action="store_true", help="Exit non-zero if CUDA is unavailable")
    args = parser.parse_args()

    data = collect()
    print("=" * 72)
    print("BIG DATA ARCHITECTURE COPILOT - GPU CHECK")
    print("=" * 72)
    print(f"PyTorch:             {data['torch_version']}")
    print(f"torch.version.cuda:  {data['torch_cuda_version']}")
    print(f"CUDA available:      {data['cuda_available']}")
    print(f"CUDA devices:        {data['cuda_device_count']}")
    if data["cuda_available"]:
        print(f"Current device:      {data['current_device']}")
        print(f"GPU:                 {data['gpu_name']}")
        print(f"Compute capability:  {data['compute_capability']}")
        print(f"VRAM:                {data['total_vram_gb']} GB")
        print(f"BF16 supported:      {data['bf16_supported']}")
        print(f"cuDNN:               {data['cudnn_version']}")
        smoke = data.get("cuda_compute_smoke", {})
        print(f"CUDA compute smoke:  {smoke.get('passed', False)}")
    print("\n--- nvidia-smi ---")
    print(data["nvidia_smi"]["stdout"] or data["nvidia_smi"]["stderr"])
    print("\n--- nvcc --version (Toolkit, if installed) ---")
    print(data["nvcc_version"]["stdout"] or data["nvcc_version"]["stderr"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nEvidence JSON saved to: {args.output}")

    gpu_ready = bool(
        data["cuda_available"]
        and data.get("cuda_compute_smoke", {}).get("passed")
        and data["nvidia_smi"].get("returncode") == 0
    )
    if args.require_cuda and not gpu_ready:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
