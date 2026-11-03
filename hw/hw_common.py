"""
hw_common.py
------------
Shared helpers for the *hardware* (PSC / vLLM) experiments. Everything that is
GPU- and vLLM-specific is funneled through here so the individual step scripts
stay small and so the pure-Python pieces (GPU->model selection, acceptance
parsing) can be unit-tested on a laptop with no GPU.

Design constraints baked in here (see FINDINGS.md for why):
  * Target hardware is a V100 (Volta, sm_70). That means **fp16 only** (no
    bf16), FlashAttention/FlashInfer are unavailable, and vLLM must be pinned
    to a Volta-capable line (<=0.18.x ships sm_70 wheels).
  * We auto-detect GPU memory and pick a target model that fits: an ~8B target
    on a 32 GB V100, a ~3B target on a 16 GB V100. Defaults are *ungated* so no
    HuggingFace approval is needed; gated Llama variants are opt-in via env.

NOTE: this module imports torch lazily. On a machine with no torch installed
(e.g. the laptop used to write/sanity-check these scripts) the pure-Python
functions still import and run; only the GPU-touching calls need torch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Make the existing controller / cost_model importable from inside hw/.
# These are the SAME objects used by the simulator -- we reuse them verbatim so
# the hardware path and the simulation path optimize an identical objective.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
@dataclass
class ModelChoice:
    """A concrete vLLM configuration sized to the detected GPU."""
    target_model: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int               # batch cap -- the sweep's upper end
    dtype: str = "float16"          # V100 has no bf16
    eagle_head: str | None = None   # EAGLE draft head (stretch goal only)
    note: str = ""

    def describe(self) -> str:
        return (f"{self.target_model} | max_len={self.max_model_len} "
                f"| max_num_seqs={self.max_num_seqs} | dtype={self.dtype} "
                f"| gpu_util={self.gpu_memory_utilization} "
                f"| eagle={self.eagle_head or '-'}  ({self.note})")


# Ungated defaults (no HF approval needed). Gated Llama variants are selected
# only when PSD_USE_LLAMA=1 AND an HF token is present.
_UNGATED_8B = "Qwen/Qwen2.5-7B-Instruct"
_UNGATED_3B = "Qwen/Qwen2.5-3B-Instruct"
_GATED_8B = "meta-llama/Llama-3.1-8B-Instruct"
_GATED_3B = "meta-llama/Llama-3.2-3B-Instruct"

# EAGLE heads matched to the target (stretch goal; only used by the EAGLE step).
# These are community EAGLE heads; confirm availability on HF before relying on
# them. They are deliberately NOT the default path.
_EAGLE_HEADS = {
    _UNGATED_8B: "duckhoi/qwen2.5-7b-eagle3.1-speculator",  # EAGLE3 head for Qwen2.5-7B
    _GATED_8B: "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
}


def pick_model(total_mem_gb: float, use_llama: bool | None = None,
               want_eagle: bool = False) -> ModelChoice:
    """
    Choose a target model + vLLM sizing from the GPU's total memory.

    Thresholds are deliberately conservative for fp16 on a V100 (weights must
    fit AND leave room for a usable KV cache so the batch-size sweep is
    meaningful):
        >= 28 GB  -> ~8B target   (32 GB V100)
        else      -> ~3B target   (16 GB V100)

    `use_llama=None` reads the PSD_USE_LLAMA env flag; set it to force.
    """
    if use_llama is None:
        use_llama = os.environ.get("PSD_USE_LLAMA", "0") == "1"
    have_hf_token = bool(os.environ.get("HF_TOKEN")
                         or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    gated_ok = use_llama and have_hf_token

    big = total_mem_gb >= 28.0
    if big:
        target = _GATED_8B if gated_ok else _UNGATED_8B
        choice = ModelChoice(
            target_model=target,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            max_num_seqs=64,
            note="8B-class target on ~32GB V100",
        )
    else:
        target = _GATED_3B if gated_ok else _UNGATED_3B
        choice = ModelChoice(
            target_model=target,
            max_model_len=4096,
            gpu_memory_utilization=0.88,
            max_num_seqs=48,
            note="3B-class target on ~16GB V100",
        )
    if want_eagle:
        choice.eagle_head = _EAGLE_HEADS.get(choice.target_model)
    return choice


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
@dataclass
class GpuInfo:
    name: str
    total_mem_gb: float
    compute_capability: tuple[int, int] | None = None
    source: str = ""                 # "torch" or "nvidia-smi"

    @property
    def is_volta(self) -> bool:
        return self.compute_capability == (7, 0)

    @property
    def supports_bf16(self) -> bool:
        # bf16 requires Ampere (sm_80) or newer.
        return (self.compute_capability is not None
                and self.compute_capability >= (8, 0))


def detect_gpu() -> GpuInfo | None:
    """
    Detect the first visible GPU. Prefer torch (gives compute capability);
    fall back to parsing `nvidia-smi`. Returns None if no GPU is visible.
    """
    # --- try torch first -------------------------------------------------
    try:
        import torch  # noqa: WPS433 (lazy import on purpose)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GpuInfo(
                name=props.name,
                total_mem_gb=props.total_memory / (1024 ** 3),
                compute_capability=(props.major, props.minor),
                source="torch",
            )
    except Exception:  # torch missing or CUDA init failed -> fall through
        pass

    # --- fall back to nvidia-smi ----------------------------------------
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if out:
            parts = [p.strip() for p in out[0].split(",")]
            name = parts[0]
            mem_gb = float(parts[1]) / 1024.0  # nvidia-smi reports MiB
            cc = None
            if len(parts) >= 3 and parts[2] and parts[2] != "[N/A]":
                maj, _, minr = parts[2].partition(".")
                cc = (int(maj), int(minr or 0))
            return GpuInfo(name=name, total_mem_gb=mem_gb,
                           compute_capability=cc, source="nvidia-smi")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Acceptance-metric parsing (from vLLM per-request spec-decode metrics)
# ---------------------------------------------------------------------------
# On Volta (V100, sm_70) vLLM's V1 engine has NO working FlashAttention,
# FlashInfer, or XFORMERS paged-decode kernel (they all require sm_80+). The
# only V1 attention backend that actually runs on V100 is vLLM's native Triton
# attention. vLLM auto-selects it on Volta, but we set it explicitly so the
# choice is logged and reproducible across every step script.
VOLTA_ATTENTION_BACKEND = "TRITON_ATTN"


def set_volta_attention_backend(gpu: "GpuInfo | None") -> None:
    """Force the V100-compatible attention backend if on Volta and unset."""
    if gpu is not None and gpu.is_volta and "VLLM_ATTENTION_BACKEND" not in os.environ:
        os.environ["VLLM_ATTENTION_BACKEND"] = VOLTA_ATTENTION_BACKEND
        print(f"Volta detected -> VLLM_ATTENTION_BACKEND={VOLTA_ATTENTION_BACKEND}")


def accepted_fraction_from_counts(num_accepted: int, num_drafted: int) -> float:
    """
    Convert raw accepted/drafted draft-token counts into the acceptance
    *fraction* in [0, 1] that controller.observe() expects.

    This is the single contract that keeps the hardware path and the simulator
    consistent: the simulator feeds `sum(accepted)/(gamma*batch)`; here we feed
    `num_accepted/num_drafted`. Both equal the fraction of proposed draft
    tokens that were accepted. We keep it a FLOAT on purpose -- see the bug
    note in controller.AdaptiveController.observe (rounding to an int biases
    the controller toward the minimum gamma).
    """
    if num_drafted <= 0:
        return 0.0
    return min(max(num_accepted / num_drafted, 0.0), 1.0)


def pretty_print_env():
    """Human-readable environment summary (used by check_env.py)."""
    gpu = detect_gpu()
    if gpu is None:
        print("No GPU detected (torch.cuda unavailable and nvidia-smi absent).")
        return None
    print(f"GPU            : {gpu.name}  (via {gpu.source})")
    print(f"Total memory   : {gpu.total_mem_gb:.1f} GB")
    print(f"Compute cap.   : {gpu.compute_capability}")
    print(f"Is Volta (V100): {gpu.is_volta}")
    print(f"bf16 supported : {gpu.supports_bf16}  "
          f"(if False, you MUST use --dtype float16)")
    return gpu


if __name__ == "__main__":
    # `python hw/hw_common.py` prints the environment + the model it would pick.
    g = pretty_print_env()
    if g is not None:
        mc = pick_model(g.total_mem_gb)
        print("\nWould run with:")
        print("  " + mc.describe())
