"""
check_env.py
------------
Step-1 environment probe, to run on the PSC GPU node BEFORE any heavy work.
Prints the GPU, compute capability, dtype support, the vLLM version, which
attention backends are importable (FlashAttention is expected to be ABSENT on
Volta -- that's fine, xFormers/Triton is the fallback), and the exact model
configuration the experiments will auto-select for this GPU.

This is the "confirm whether vLLM imports / tell me the GPU" gate from the
project plan. It touches the GPU only through torch; it does NOT load a model,
so it is fast and safe to run first.
"""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import detect_gpu, pick_model  # noqa: E402


def _probe_backend(modname: str) -> str:
    try:
        importlib.import_module(modname)
        return "available"
    except Exception as e:
        return f"absent ({type(e).__name__})"


def main() -> int:
    print("=" * 64)
    print("STEP 1 ENVIRONMENT PROBE")
    print("=" * 64)

    gpu = detect_gpu()
    if gpu is None:
        print("\nNO GPU DETECTED.")
        print("torch.cuda is unavailable and nvidia-smi returned nothing.")
        print("Per the project's hard rule: STOP. Run this on a GPU node")
        print("(inside salloc/srun on PSC), not a login node.")
        return 1

    print(f"\nGPU             : {gpu.name}  (detected via {gpu.source})")
    print(f"Total memory    : {gpu.total_mem_gb:.1f} GB")
    print(f"Compute capab.  : {gpu.compute_capability}")
    print(f"Volta / sm_70   : {gpu.is_volta}")
    print(f"bf16 supported  : {gpu.supports_bf16}")
    if not gpu.supports_bf16:
        print("  -> MUST pass dtype=float16 everywhere (handled automatically).")

    print("\n--- torch ---")
    try:
        import torch
        print(f"torch           : {torch.__version__} (cuda {torch.version.cuda})")
        print(f"arch list       : {getattr(torch.cuda, 'get_arch_list', lambda: [])()}")
    except Exception as e:
        print(f"torch import FAILED: {e}")
        return 2

    print("\n--- vLLM ---")
    try:
        import vllm
        print(f"vllm            : {vllm.__version__}")
    except Exception as e:
        print(f"vllm import FAILED: {e}")
        print("Install it with: bash hw/setup_psc.sh")
        return 3

    print("\n--- attention backends (FlashAttn expected ABSENT on V100) ---")
    print(f"flash_attn      : {_probe_backend('flash_attn')}")
    print(f"flashinfer      : {_probe_backend('flashinfer')}")
    print(f"xformers        : {_probe_backend('xformers')}")

    print("\n--- auto-selected run configuration ---")
    mc = pick_model(gpu.total_mem_gb)
    print("  " + mc.describe())
    print("\nVLLM_ATTENTION_BACKEND currently =",
          os.environ.get("VLLM_ATTENTION_BACKEND", "(unset -> vLLM auto-picks)"))
    if gpu.is_volta:
        print("On V100, if a run errors about FlashAttention/FlashInfer, set:")
        print("    export VLLM_ATTENTION_BACKEND=XFORMERS")

    print("\nENV PROBE OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
