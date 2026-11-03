"""
step3_calibrate.py
------------------
Step 3: calibrate cost_model.HardwareModel against real decode microbenchmarks
on this GPU, replacing the guessed t_mem / t_flop constants with measured ones.

Method (clean, prefill-free):
  The target model's decode-step latency under the roofline model is
      t_step(B) = t_mem_target + t_flop_target * B
  (one position decoded per request, so n_tokens == batch B). We measure it
  WITHOUT speculative decoding by, for each batch size B, running B identical
  prompts twice -- once generating N1 tokens, once N2 (> N1) -- and taking
      t_step(B) = (wall(N2) - wall(N1)) / (N2 - N1).
  The subtraction cancels the prefill and all fixed per-call overhead, leaving
  the pure per-decode-step cost. A linear fit of t_step(B) vs B then gives
  t_mem_target (intercept) and t_flop_target (slope).

For n-gram speculation the "draft" is a CPU table lookup with no target-model
FLOPs, so the draft terms are set to ~0; the macro-step cost is dominated by
the single target verification pass over B*(gamma+1) positions, which the
fitted target constants already predict. (For the EAGLE stretch goal, the draft
terms would need their own calibration.)

Writes results_hw/calibrated_constants.json, consumed by
cost_model.HardwareModel.load_calibrated() in BOTH the vLLM path and the
simulator, so the two stay consistent.

Run (GPU node, venv active):
    python hw/step3_calibrate.py --batches 1,2,4,8,16,32,48,64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import detect_gpu, pick_model, set_volta_attention_backend  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost_model import HardwareModel  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_hw")


def _time_generate(llm, sp, prompts) -> float:
    t = time.time()
    llm.generate(prompts, sp)
    return time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32,48,64")
    ap.add_argument("--n1", type=int, default=32, help="short token count")
    ap.add_argument("--n2", type=int, default=160, help="long token count")
    ap.add_argument("--eager", action="store_true", default=True)
    ap.add_argument("--no-eager", dest="eager", action="store_false")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU -- STOP (run on a GPU node).")
        return 1
    set_volta_attention_backend(gpu)
    os.makedirs(RES, exist_ok=True)

    mc = pick_model(gpu.total_mem_gb)
    batches = [b for b in batches if b <= mc.max_num_seqs]
    print(f"GPU: {gpu.name} ({gpu.total_mem_gb:.0f} GB)")
    print(f"Model: {mc.target_model} | calibration batches={batches} | "
          f"N1={args.n1} N2={args.n2}")

    from vllm import LLM, SamplingParams

    # Plain decode (NO speculative_config); disable prefix caching so timing is
    # clean and reproducible across the two token counts.
    t0 = time.time()
    llm = LLM(
        model=mc.target_model,
        dtype=mc.dtype,
        max_model_len=mc.max_model_len,
        gpu_memory_utilization=mc.gpu_memory_utilization,
        max_num_seqs=mc.max_num_seqs,
        trust_remote_code=True,
        enforce_eager=args.eager,
        enable_prefix_caching=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # A single fixed prompt, repeated B times -> a batch of B decode streams.
    prompt = ("Write a long, detailed story that keeps going without stopping. "
              "Begin now:")
    sp1 = SamplingParams(temperature=0.0, max_tokens=args.n1, ignore_eos=True)
    sp2 = SamplingParams(temperature=0.0, max_tokens=args.n2, ignore_eos=True)

    # Warm up.
    print("Warming up...")
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8,
                                          ignore_eos=True))

    rows = []
    for B in batches:
        prompts = [prompt] * B
        t_short = _time_generate(llm, sp1, prompts)
        t_long = _time_generate(llm, sp2, prompts)
        step_lat = (t_long - t_short) / (args.n2 - args.n1)  # seconds / step
        rows.append({"batch": B, "t_short_s": round(t_short, 4),
                     "t_long_s": round(t_long, 4),
                     "step_latency_s": step_lat})
        print(f"  B={B:>3} | t({args.n1})={t_short:6.2f}s  "
              f"t({args.n2})={t_long:6.2f}s  -> step={step_lat*1e3:7.3f} ms")

    # --- linear fit: step_latency = t_mem + t_flop * B ---
    Bs = np.array([r["batch"] for r in rows], dtype=float)
    lat = np.array([r["step_latency_s"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(Bs, lat, 1)      # t_flop, t_mem
    pred = slope * Bs + intercept
    ss_res = float(np.sum((lat - pred) ** 2))
    ss_tot = float(np.sum((lat - lat.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    t_mem_target = float(max(intercept, 0.0))
    t_flop_target = float(max(slope, 0.0))

    defaults = HardwareModel()
    print("\n--- calibration result ---")
    print(f"  t_mem_target : {t_mem_target*1e3:.4f} ms   "
          f"(was {defaults.t_mem_target*1e3:.4f} ms)")
    print(f"  t_flop_target: {t_flop_target*1e3:.6f} ms/token   "
          f"(was {defaults.t_flop_target*1e3:.6f} ms/token)")
    print(f"  fit R^2      : {r2:.4f}")
    print("  t_mem_draft / t_flop_draft set to 0 (n-gram draft = CPU lookup, "
          "no target FLOPs)")

    out = {
        "gpu": gpu.name,
        "model": mc.target_model,
        "method": "ngram",
        "fit_r2": r2,
        "n1": args.n1, "n2": args.n2,
        "constants": {
            "t_mem_target": t_mem_target,
            "t_flop_target": t_flop_target,
            "t_mem_draft": 0.0,
            "t_flop_draft": 0.0,
        },
        "raw": rows,
    }
    path = os.path.join(RES, "calibrated_constants.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")

    # Sanity check: the loader should now return the fitted constants.
    hw = HardwareModel.load_calibrated(path)
    assert abs(hw.t_mem_target - t_mem_target) < 1e-12
    print("Verified: cost_model.HardwareModel.load_calibrated() picks these up.")
    print("STEP 3 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
