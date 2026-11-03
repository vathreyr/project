"""
step2_baseline.py
-----------------
Step 2: run vLLM speculative decoding (n-gram) at a FIXED speculation length
and log, across a batch-size sweep, the realized acceptance and timing that the
controller and cost model depend on:

    batch size B, acceptance fraction, mean acceptance length, throughput,
    and mean decode-step latency.

Everything is MEASURED on the GPU via vLLM's own metrics (llm.get_metrics(),
the vllm:spec_decode_* counters) -- no fabricated numbers. Results go to
results_hw/ as CSV + a raw JSON dump.

Why offline (not a server): a single offline LLM lets us hold gamma fixed, vary
the batch size deterministically by submitting B prompts at once, and read the
exact accepted/drafted token counts by diffing the cumulative counters before
and after each batch. That is the cleanest way to characterize acceptance and
step latency vs load, and it feeds step3 (cost-model calibration) directly.

Run (GPU node, venv active):
    python hw/step2_baseline.py                 # defaults: gamma=4, sweep
    python hw/step2_baseline.py --gamma 4 --batches 1,4,16,32,64 --max-tokens 128
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import (detect_gpu, pick_model, set_volta_attention_backend,  # noqa: E402
                       accepted_fraction_from_counts)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_hw")

# vLLM spec-decode counter names (confirmed against the v0.11.0 offline example).
_M_DRAFTS = "vllm:spec_decode_num_drafts"
_M_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
_M_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"


def _sum_metric(metrics, name: str) -> float:
    """Sum the value(s) of a named metric from llm.get_metrics() output.

    get_metrics() returns metric objects with a `.name` and either a scalar
    `.value` (Counter) or a `.values` list (Vector); we handle both defensively
    since the exact shape varies across vLLM point releases.
    """
    total = 0.0
    for m in metrics:
        if getattr(m, "name", None) != name:
            continue
        if hasattr(m, "value") and m.value is not None:
            total += float(m.value)
        elif hasattr(m, "values") and m.values is not None:
            total += float(sum(m.values))
    return total


def _snapshot(llm) -> dict:
    """Cumulative spec-decode counters right now (empty if unsupported)."""
    try:
        metrics = llm.get_metrics()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] get_metrics() failed: {e}")
        return {}
    return {
        "drafts": _sum_metric(metrics, _M_DRAFTS),
        "draft_tokens": _sum_metric(metrics, _M_DRAFT_TOKENS),
        "accepted_tokens": _sum_metric(metrics, _M_ACCEPTED_TOKENS),
    }


def _make_prompts(n: int) -> list[str]:
    """
    A fixed pool of prompts, cycled to length n. We mix in prompts that ask the
    model to continue/expand structured text, so n-gram speculation (which only
    accepts tokens that repeat an n-gram already in the context) has realistic,
    non-trivial acceptance to measure. The exact acceptance is whatever the
    model produces -- we report it, we do not engineer it.
    """
    pool = [
        "List the first 20 prime numbers, separated by commas.",
        "Repeat the following sentence five times: The quick brown fox jumps "
        "over the lazy dog.",
        "Summarize this text, then repeat the summary verbatim: Speculative "
        "decoding proposes draft tokens that the target model verifies in a "
        "single pass, accepting the longest correct prefix.",
        "Write the numbers 1 through 30, each on its own line.",
        "Explain continuous batching in LLM serving, then restate your "
        "explanation in the same words.",
        "Continue this list of US states: Alabama, Alaska, Arizona, Arkansas,",
    ]
    return [pool[i % len(pool)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=int, default=4,
                    help="fixed speculation length (num_speculative_tokens)")
    ap.add_argument("--batches", type=str, default="1,4,8,16,32,48,64",
                    help="comma-separated batch sizes to sweep")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-lookup-max", type=int, default=5)
    ap.add_argument("--prompt-lookup-min", type=int, default=2)
    ap.add_argument("--eager", action="store_true", default=True,
                    help="enforce_eager (default on: avoids spec-decode cudagraph "
                         "issues on Volta and loads faster)")
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
    # Cap the sweep at the model's configured batch ceiling.
    batches = [b for b in batches if b <= mc.max_num_seqs]
    print(f"GPU: {gpu.name} ({gpu.total_mem_gb:.0f} GB)")
    print(f"Model: {mc.target_model}  | gamma={args.gamma}  | batches={batches}")

    from vllm import LLM, SamplingParams

    spec_cfg = {
        "method": "ngram",
        "num_speculative_tokens": args.gamma,
        "prompt_lookup_max": args.prompt_lookup_max,
        "prompt_lookup_min": args.prompt_lookup_min,
    }

    t0 = time.time()
    llm = LLM(
        model=mc.target_model,
        dtype=mc.dtype,
        max_model_len=mc.max_model_len,
        gpu_memory_utilization=mc.gpu_memory_utilization,
        max_num_seqs=mc.max_num_seqs,
        trust_remote_code=True,
        enforce_eager=args.eager,
        speculative_config=spec_cfg,
        disable_log_stats=False,     # needed for get_metrics()
    )
    print(f"Model+spec loaded in {time.time() - t0:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Warm up once so the first measured batch isn't penalized by lazy init /
    # first-call overhead (this is why an un-warmed B=1 looks artificially slow).
    print("Warming up...")
    llm.generate(["Hello, warm up the engine."],
                 SamplingParams(temperature=0.0, max_tokens=8))

    rows = []
    prev = _snapshot(llm)
    metrics_supported = bool(prev)
    if not metrics_supported:
        print("  [warn] spec-decode metrics unavailable; will log timing only.")

    for B in batches:
        prompts = _make_prompts(B)
        # Warm nothing special; measure the whole generate() call.
        t_start = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t_start

        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        now = _snapshot(llm)

        if metrics_supported and now:
            d_drafts = now["drafts"] - prev["drafts"]
            d_draft_tok = now["draft_tokens"] - prev["draft_tokens"]
            d_acc_tok = now["accepted_tokens"] - prev["accepted_tokens"]
            prev = now
            accept_frac = accepted_fraction_from_counts(int(d_acc_tok),
                                                         int(d_draft_tok))
            accept_len = 1.0 + (d_acc_tok / d_drafts) if d_drafts > 0 else 1.0
        else:
            d_drafts = d_draft_tok = d_acc_tok = 0
            accept_frac = accept_len = float("nan")

        throughput = out_tokens / dt if dt > 0 else 0.0
        # Mean wall time per output token across the batch (TPOT-like proxy).
        tpot = dt / (out_tokens / B) if out_tokens > 0 else float("nan")

        row = {
            "gamma": args.gamma,
            "batch": B,
            "wall_s": round(dt, 4),
            "out_tokens": out_tokens,
            "throughput_tok_s": round(throughput, 2),
            "tpot_s": round(tpot, 5),
            "accept_fraction": round(accept_frac, 4),
            "accept_length": round(accept_len, 4),
            "num_drafts": int(d_drafts),
            "draft_tokens": int(d_draft_tok),
            "accepted_tokens": int(d_acc_tok),
        }
        rows.append(row)
        print(f"  B={B:>3} | thr={throughput:7.1f} tok/s | "
              f"accept_frac={accept_frac:.3f} | accept_len={accept_len:.2f} | "
              f"wall={dt:.2f}s")

    # --- write outputs ---
    csv_path = os.path.join(RES, f"step2_baseline_g{args.gamma}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = os.path.join(RES, f"step2_baseline_g{args.gamma}.json")
    # Rebuild a clean spec-config dict: vLLM mutates the one we passed in,
    # inserting non-serializable ModelConfig objects. default=str is a belt-and
    # -suspenders guard for anything else.
    clean_spec = {"method": "ngram", "num_speculative_tokens": args.gamma,
                  "prompt_lookup_max": args.prompt_lookup_max,
                  "prompt_lookup_min": args.prompt_lookup_min}
    with open(json_path, "w") as f:
        json.dump({"gpu": gpu.name, "model": mc.target_model,
                   "gamma": args.gamma, "spec_config": clean_spec,
                   "metrics_supported": metrics_supported, "rows": rows}, f,
                  indent=2, default=str)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print("STEP 2 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
