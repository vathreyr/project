"""
probe_eagle.py
--------------
Decisive ~5-minute probe before investing in the EAGLE path. Answers two
questions that determine how the rest is built, and prints a clear verdict:

  (1) Does THIS vLLM build's SpeculativeConfig accept
      `num_speculative_tokens_per_batch_size` (the native batch->gamma schedule
      that would let a stock server be load-adaptive without a patch)?
  (2) Does EAGLE3 for Qwen2.5-7B actually LOAD and GENERATE on this V100?
      EAGLE is the least Volta-portable spec-decode method; this is the thing
      most likely to fail on sm_70.

It loads the EAGLE draft, runs one short generation, and reports acceptance
from vLLM's own metrics. If it crashes, the traceback tells us whether EAGLE is
simply unsupported on Volta (then we document that honestly) or needs a tweak.

Run (GPU node, venv active):
    python hw/probe_eagle.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import detect_gpu, pick_model, set_volta_attention_backend  # noqa: E402


def check_batch_size_schedule() -> bool:
    """Introspect SpeculativeConfig for the batch-size schedule field."""
    try:
        from vllm.config import SpeculativeConfig
        fields = getattr(SpeculativeConfig, "__dataclass_fields__", {})
        has = "num_speculative_tokens_per_batch_size" in fields
        print(f"[1] num_speculative_tokens_per_batch_size supported: {has}")
        return has
    except Exception as e:  # noqa: BLE001
        print(f"[1] could not introspect SpeculativeConfig: {e}")
        return False


def try_eagle() -> bool:
    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU -- STOP.")
        return False
    set_volta_attention_backend(gpu)
    mc = pick_model(gpu.total_mem_gb, want_eagle=True)
    if not mc.eagle_head:
        print(f"[2] no EAGLE head configured for {mc.target_model}; skipping.")
        return False
    print(f"[2] attempting EAGLE3: target={mc.target_model} "
          f"draft={mc.eagle_head}")

    from vllm import LLM, SamplingParams
    spec_cfg = {
        "method": "eagle3",
        "model": mc.eagle_head,
        "num_speculative_tokens": 3,
    }
    try:
        t0 = time.time()
        llm = LLM(
            model=mc.target_model,
            dtype=mc.dtype,
            max_model_len=2048,
            gpu_memory_utilization=mc.gpu_memory_utilization,
            max_num_seqs=8,
            trust_remote_code=True,
            enforce_eager=True,
            speculative_config=spec_cfg,
            disable_log_stats=False,
        )
        print(f"[2] EAGLE LLM constructed in {time.time()-t0:.1f}s")
        outs = llm.generate(["Explain speculative decoding in two sentences."],
                            SamplingParams(temperature=0.0, max_tokens=64))
        txt = outs[0].outputs[0].text.strip()[:200]
        print(f"[2] EAGLE generation OK: {txt!r}")
        # Acceptance from metrics, if exposed.
        try:
            acc = dr = 0.0
            for m in llm.get_metrics():
                if m.name == "vllm:spec_decode_num_accepted_tokens":
                    acc += float(getattr(m, "value", 0) or 0)
                elif m.name == "vllm:spec_decode_num_draft_tokens":
                    dr += float(getattr(m, "value", 0) or 0)
            if dr > 0:
                print(f"[2] EAGLE acceptance fraction: {acc/dr:.3f}  "
                      f"(accepted {int(acc)} / drafted {int(dr)})")
        except Exception:  # noqa: BLE001
            pass
        print("\nVERDICT: EAGLE RUNS ON THIS V100. ✅  Build the EAGLE sweep.")
        return True
    except Exception:  # noqa: BLE001
        print("\n[2] EAGLE FAILED to load/generate on this V100:")
        traceback.print_exc()
        print("\nVERDICT: EAGLE does NOT run here (likely a Volta/sm_70 kernel "
              "limitation). We document this honestly and keep n-gram as the "
              "primary; the goodput sweep does not depend on EAGLE.")
        return False


if __name__ == "__main__":
    print("=" * 64)
    check_batch_size_schedule()
    ok = try_eagle()
    print("=" * 64)
    raise SystemExit(0 if ok else 2)
