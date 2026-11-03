"""
step4_grid.py
-------------
Step 4 (measurement half): sweep one speculation length gamma over a range of
batch sizes and record REAL throughput + acceptance, appending to a shared grid
CSV. Run it once per gamma (including gamma=0 = no speculative decoding); a
fresh process per gamma guarantees clean GPU memory between model loads.

    for g in 0 1 2 4 6; do python hw/step4_grid.py --gamma $g; done

The resulting results_hw/step4_grid.csv is the (gamma x batch) grid that
step5_policies.py turns into the policy comparison (no-spec / static / adaptive
/ oracle). The adaptive policy's gamma(batch) choice is made by the SAME
controller.AdaptiveController objective used in the simulator, fed the
acceptance measured here -- so sim and hardware stay consistent.

gamma=0 builds a plain LLM (no speculative_config); gamma>0 uses n-gram.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import (detect_gpu, pick_model, set_volta_attention_backend,  # noqa: E402
                       accepted_fraction_from_counts)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_hw")
GRID_CSV = os.path.join(RES, "step4_grid.csv")

_M_DRAFTS = "vllm:spec_decode_num_drafts"
_M_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
_M_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"

_FIELDS = ["gamma", "batch", "wall_s", "out_tokens", "throughput_tok_s",
           "tpot_s", "accept_fraction", "accept_length",
           "num_drafts", "draft_tokens", "accepted_tokens"]


def _sum_metric(metrics, name):
    total = 0.0
    for m in metrics:
        if getattr(m, "name", None) != name:
            continue
        if getattr(m, "value", None) is not None:
            total += float(m.value)
        elif getattr(m, "values", None) is not None:
            total += float(sum(m.values))
    return total


def _snapshot(llm):
    try:
        ms = llm.get_metrics()
    except Exception:  # noqa: BLE001
        return {}
    return {"drafts": _sum_metric(ms, _M_DRAFTS),
            "draft_tokens": _sum_metric(ms, _M_DRAFT_TOKENS),
            "accepted_tokens": _sum_metric(ms, _M_ACCEPTED_TOKENS)}


def _prompts(n):
    pool = [
        "List the first 20 prime numbers, separated by commas.",
        "Repeat this sentence five times: The quick brown fox jumps over the "
        "lazy dog.",
        "Summarize then repeat verbatim: speculative decoding proposes draft "
        "tokens the target verifies in one pass.",
        "Write the numbers 1 through 30, each on its own line.",
        "Explain continuous batching, then restate it in the same words.",
        "Continue this list of US states: Alabama, Alaska, Arizona, Arkansas,",
    ]
    return [pool[i % len(pool)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=int, required=True,
                    help="speculation length; 0 = no spec decode (plain)")
    ap.add_argument("--method", choices=["ngram", "eagle"], default="ngram",
                    help="spec-decode method (eagle needs a draft head + Ampere+)")
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32,48,64")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-lookup-max", type=int, default=5)
    ap.add_argument("--prompt-lookup-min", type=int, default=2)
    ap.add_argument("--fresh", action="store_true",
                    help="overwrite the grid CSV instead of appending")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU -- STOP.")
        return 1
    set_volta_attention_backend(gpu)
    os.makedirs(RES, exist_ok=True)

    want_eagle = args.method == "eagle"
    mc = pick_model(gpu.total_mem_gb, want_eagle=want_eagle)
    if want_eagle and args.gamma > 0 and not mc.eagle_head:
        print(f"NO EAGLE head configured for {mc.target_model}; STOP.")
        return 1
    batches = [b for b in batches if b <= mc.max_num_seqs]
    print(f"GPU: {gpu.name} | model: {mc.target_model} | method={args.method} | "
          f"gamma={args.gamma} | batches={batches}")

    from vllm import LLM, SamplingParams

    kw = dict(model=mc.target_model, dtype=mc.dtype,
              max_model_len=mc.max_model_len,
              gpu_memory_utilization=mc.gpu_memory_utilization,
              max_num_seqs=mc.max_num_seqs, trust_remote_code=True,
              enforce_eager=True, disable_log_stats=False)
    if args.gamma > 0:
        if want_eagle:
            kw["speculative_config"] = {
                "method": "eagle3",
                "model": mc.eagle_head,
                "num_speculative_tokens": args.gamma,
            }
        else:
            kw["speculative_config"] = {
                "method": "ngram",
                "num_speculative_tokens": args.gamma,
                "prompt_lookup_max": args.prompt_lookup_max,
                "prompt_lookup_min": args.prompt_lookup_min,
            }

    t0 = time.time()
    llm = LLM(**kw)
    print(f"Loaded in {time.time() - t0:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print("Warming up...")
    llm.generate(["Hello, warm up."],
                 SamplingParams(temperature=0.0, max_tokens=8))

    rows = []
    prev = _snapshot(llm)
    spec_metrics_ok = bool(prev) and args.gamma > 0

    for B in batches:
        prompts = _prompts(B)
        t_start = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t_start
        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

        if spec_metrics_ok:
            now = _snapshot(llm)
            d_dr = now["drafts"] - prev["drafts"]
            d_dt = now["draft_tokens"] - prev["draft_tokens"]
            d_at = now["accepted_tokens"] - prev["accepted_tokens"]
            prev = now
            acc_frac = accepted_fraction_from_counts(int(d_at), int(d_dt))
            acc_len = 1.0 + (d_at / d_dr) if d_dr > 0 else 1.0
        else:
            d_dr = d_dt = d_at = 0
            acc_frac = 0.0          # no-spec: no draft tokens
            acc_len = 1.0

        thr = out_tokens / dt if dt > 0 else 0.0
        tpot = dt / (out_tokens / B) if out_tokens > 0 else float("nan")
        rows.append({"gamma": args.gamma, "batch": B, "wall_s": round(dt, 4),
                     "out_tokens": out_tokens, "throughput_tok_s": round(thr, 2),
                     "tpot_s": round(tpot, 6), "accept_fraction": round(acc_frac, 4),
                     "accept_length": round(acc_len, 4), "num_drafts": int(d_dr),
                     "draft_tokens": int(d_dt), "accepted_tokens": int(d_at)})
        print(f"  B={B:>3} | thr={thr:7.1f} tok/s | acc_frac={acc_frac:.3f} | "
              f"acc_len={acc_len:.2f}")

    # Method-specific grid: n-gram keeps the original name (backward-compatible
    # with the V100 data); EAGLE goes to its own file.
    grid_csv = GRID_CSV if args.method == "ngram" else \
        os.path.join(RES, "step4_grid_eagle.csv")
    write_header = args.fresh or not os.path.exists(grid_csv)
    mode = "w" if args.fresh else "a"
    with open(grid_csv, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\n{'Wrote' if write_header else 'Appended to'} {grid_csv}")
    print(f"STEP 4 (method={args.method}, gamma={args.gamma}) DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
