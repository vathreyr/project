"""
step1_smoke.py
--------------
Step 1 (final gate): confirm plain vLLM generation works end-to-end on this
GPU, with NO speculative decoding. If this does not run, nothing later will, so
we stop here and fix the environment first.

It auto-detects the GPU, picks a fitting fp16 target model, loads it offline,
generates a handful of prompts, and reports tokens/sec. On a Volta GPU it sets
the xFormers attention backend automatically (FlashAttention is unavailable).

Run (on a GPU node, venv activated):
    python hw/step1_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import detect_gpu, pick_model, set_volta_attention_backend  # noqa: E402


def main() -> int:
    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU -- STOP (run on a GPU node). See hard rule in the plan.")
        return 1

    # Volta (V100) needs vLLM's Triton attention backend; xFormers/FlashAttn
    # have no sm_70 paged-decode kernel. Centralized in hw_common.
    set_volta_attention_backend(gpu)

    mc = pick_model(gpu.total_mem_gb)
    print(f"GPU: {gpu.name} ({gpu.total_mem_gb:.0f} GB)")
    print(f"Loading (no spec decode): {mc.describe()}")

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=mc.target_model,
        dtype=mc.dtype,
        max_model_len=mc.max_model_len,
        gpu_memory_utilization=mc.gpu_memory_utilization,
        max_num_seqs=mc.max_num_seqs,
        trust_remote_code=True,
        enforce_eager=True,   # skip the ~80s torch.compile for a fast smoke test
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    prompts = [
        "Explain what speculative decoding is in two sentences.",
        "Write a haiku about GPUs.",
        "List three reasons batching helps LLM serving throughput.",
        "What is the capital of France, and why is it famous?",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\nGenerated {total_out_tokens} tokens in {dt:.2f}s "
          f"= {total_out_tokens / dt:.1f} tok/s (batch of {len(prompts)})")
    print("\n--- sample output ---")
    print(outs[0].outputs[0].text.strip()[:400])
    print("\nSTEP 1 SMOKE TEST PASSED: plain vLLM generation works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
