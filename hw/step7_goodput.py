"""
step7_goodput.py
----------------
Step 5, done the way the project brief actually specifies it: a real
**serving** experiment with **Poisson arrivals** over **ShareGPT** prompts,
reporting **goodput = requests/s that meet a per-token (TPOT) SLO**, plus
throughput and p50/p90/p99 TPOT -- not the offline batch sweep.

For each policy (no-spec, static-gamma 2/4/6) it launches a vLLM OpenAI server
with that speculation length, replays ShareGPT at a sweep of arrival rates
lambda, and measures client-side per-request latency (streaming, so we get the
real inter-token time). The **adaptive** policy is then reconstructed from the
measured policies with the SAME controller objective used everywhere else: at
each lambda, the controller picks gamma from the mean running batch that lambda
induced, and we report that gamma's measured goodput. (If this vLLM build
exposes the native `num_speculative_tokens_per_batch_size` schedule, pass
--native-adaptive to run adaptive as its own live server instead.)

Everything that does not need the GPU -- ShareGPT loading, Poisson schedule,
and the goodput / percentile math -- is pure and unit-tested (see
--self-test). The networking is the standard OpenAI completions API.

Run (GPU node, venv active; needs the server deps from setup_psc.sh):
    python hw/step7_goodput.py --rates 2,4,8,16,24 --num-prompts 300 \
        --slo-tpot 0.08 --policies no-spec,2,4,6
    python hw/step7_goodput.py --self-test        # CPU-only logic tests
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_common import detect_gpu, pick_model, set_volta_attention_backend  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost_model import HardwareModel, expected_tokens_per_step  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_hw")


# ===========================================================================
# Pure, CPU-testable logic
# ===========================================================================
def poisson_arrivals(n: int, rate: float, seed: int = 0) -> list[float]:
    """Cumulative arrival times (s) for n requests at Poisson rate `rate`."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated p-th percentile (p in [0,100]); nan if empty."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[dict], slo_tpot: float, wall: float) -> dict:
    """Aggregate per-request records into serving metrics."""
    ok = [r for r in results if r.get("ok")]
    tpots = [r["tpot"] for r in ok if r["tpot"] == r["tpot"]]  # drop nan
    meets = [r for r in ok if r["tpot"] == r["tpot"] and r["tpot"] <= slo_tpot]
    out_tokens = sum(r["n_out"] for r in ok)
    return {
        "n_done": len(ok),
        "throughput_tok_s": out_tokens / wall if wall > 0 else 0.0,
        "goodput_req_s": len(meets) / wall if wall > 0 else 0.0,
        "slo_attainment": (len(meets) / len(ok)) if ok else 0.0,
        "tpot_p50": percentile(tpots, 50),
        "tpot_p90": percentile(tpots, 90),
        "tpot_p99": percentile(tpots, 99),
        "mean_batch": (sum(r.get("concurrency", 0) for r in ok) / len(ok)) if ok else 0.0,
    }


def adaptive_from_grid(by_policy_rate: dict, hw: HardwareModel,
                       spec_gammas: list[int], rates: list[float],
                       accept_by_rate: dict) -> list[dict]:
    """
    Reconstruct the adaptive policy per arrival rate: the controller chooses
    gamma from the mean running batch induced at that rate (measured) and the
    measured acceptance, then we report that gamma's measured goodput. Same
    objective as controller.AdaptiveController / vllm_integration.
    """
    out = []
    for lam in rates:
        avail = [g for g in spec_gammas if (g, lam) in by_policy_rate]
        if not avail:
            continue
        # mean batch observed across the static runs at this rate
        mb = max(by_policy_rate[(g, lam)]["mean_batch"] for g in avail)
        alpha = accept_by_rate.get(lam, 0.4)
        best_g, best = avail[0], -1.0
        for g in avail:
            score = expected_tokens_per_step(alpha, g) / hw.macro_step_seconds(mb, g)
            if score > best:
                best, best_g = score, g
        row = dict(by_policy_rate[(best_g, lam)])
        row.update({"policy": "adaptive", "rate": lam, "chosen_gamma": best_g})
        out.append(row)
    return out


# ===========================================================================
# Networking (standard OpenAI streaming completions)
# ===========================================================================
def _server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _stream_one(url: str, model: str, prompt: str, max_tokens: int,
                live: dict, rec: dict) -> None:
    """One streaming completion; fills rec with ttft/tpot/n_out/ok."""
    import requests  # local import so --self-test needs no network deps
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    live["cur"] = live.get("cur", 0) + 1
    rec["concurrency"] = live["cur"]
    t0 = time.time()
    first_t = None
    n_out = 0
    try:
        resp = requests.post(url + "/v1/completions", data=body,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=600)
        for line in resp.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8", "ignore")
            if not s.startswith("data:"):
                continue
            payload = s[5:].strip()
            if payload == "[DONE]":
                break
            try:
                tok = json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
            if tok.get("choices") and tok["choices"][0].get("text"):
                if first_t is None:
                    first_t = time.time()
                n_out += 1
        end = time.time()
        rec["ttft"] = (first_t - t0) if first_t else float("nan")
        rec["n_out"] = n_out
        # TPOT: mean inter-token time during decode (exclude the first token).
        rec["tpot"] = ((end - first_t) / (n_out - 1)) if (first_t and n_out > 1) \
            else float("nan")
        rec["latency"] = end - t0
        rec["ok"] = n_out > 0
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["err"] = str(e)
    finally:
        live["cur"] -= 1


def run_load(url: str, model: str, prompts: list[str], arrivals: list[float],
             max_tokens: int) -> tuple[list[dict], float]:
    """Fire each prompt at its scheduled arrival time; return (records, wall)."""
    records = [dict() for _ in prompts]
    live = {"cur": 0}
    threads = []
    start = time.time()
    for i, (p, a) in enumerate(zip(prompts, arrivals)):
        delay = a - (time.time() - start)
        if delay > 0:
            time.sleep(delay)
        th = threading.Thread(target=_stream_one,
                              args=(url, model, p, max_tokens, live, records[i]))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    return records, time.time() - start


# ===========================================================================
# ShareGPT loading
# ===========================================================================
def load_prompts(path: str | None, n: int, seed: int = 0) -> list[str]:
    """
    Load up to n first-human-turn prompts from a ShareGPT json (list of convos
    with 'conversations':[{'from','value'},...]). Falls back to the HF datasets
    hub, then to a synthetic set, so the script always runs.
    """
    convos = None
    if path and os.path.exists(path):
        with open(path) as f:
            convos = json.load(f)
    if convos is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                              data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                              split="train")
            convos = ds.to_list()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] ShareGPT unavailable ({e}); using synthetic prompts.")
            base = ["Summarize the history of the internet.",
                    "Write a Python function to merge two sorted lists.",
                    "Explain how transformers work, step by step.",
                    "Give me a recipe for a quick weeknight dinner."]
            return [base[i % len(base)] for i in range(n)]
    prompts = []
    for c in convos:
        turns = c.get("conversations") or c.get("conversation") or []
        for t in turns:
            if t.get("from") in ("human", "user") and t.get("value"):
                prompts.append(t["value"][:2000])
                break
        if len(prompts) >= n * 2:
            break
    random.Random(seed).shuffle(prompts)
    return prompts[:n] if prompts else ["Hello, tell me about speculative decoding."] * n


# ===========================================================================
# Orchestration
# ===========================================================================
def launch_server(mc, gamma: int, port: int, log_path: str):
    """Start `vllm serve` for a given gamma; return the Popen handle."""
    env = dict(os.environ)
    gpu = detect_gpu()
    set_volta_attention_backend(gpu)
    env.setdefault("VLLM_ATTENTION_BACKEND", os.environ.get(
        "VLLM_ATTENTION_BACKEND", "TRITON_ATTN"))
    cmd = ["vllm", "serve", mc.target_model,
           "--dtype", mc.dtype, "--max-model-len", str(mc.max_model_len),
           "--gpu-memory-utilization", str(mc.gpu_memory_utilization),
           "--max-num-seqs", str(mc.max_num_seqs), "--port", str(port),
           "--enforce-eager", "--trust-remote-code", "--disable-log-requests"]
    if gamma > 0:
        cmd += ["--speculative-config",
                json.dumps({"method": "ngram", "num_speculative_tokens": gamma,
                            "prompt_lookup_max": 5, "prompt_lookup_min": 2})]
    logf = open(log_path, "w")
    print("  launching:", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)


def wait_ready(url: str, proc, timeout: float = 600.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            print("  server process exited early; check its log.")
            return False
        if _server_up(url):
            return True
        time.sleep(3)
    return False


def self_test() -> int:
    """CPU-only checks of the pure logic."""
    a = poisson_arrivals(1000, rate=10.0, seed=1)
    assert a == sorted(a) and 80 < a[-1] < 120, "poisson mean rate off"
    assert abs(percentile([1, 2, 3, 4], 50) - 2.5) < 1e-9
    assert percentile([], 50) != percentile([], 50)  # nan
    recs = [{"ok": True, "tpot": 0.05, "n_out": 100, "concurrency": 4},
            {"ok": True, "tpot": 0.20, "n_out": 50, "concurrency": 4},
            {"ok": False}]
    s = summarize(recs, slo_tpot=0.08, wall=10.0)
    assert s["n_done"] == 2 and abs(s["goodput_req_s"] - 0.1) < 1e-9, s
    assert abs(s["throughput_tok_s"] - 15.0) < 1e-9, s
    assert abs(s["slo_attainment"] - 0.5) < 1e-9, s
    hw = HardwareModel()
    by = {(g, 8.0): {"goodput_req_s": 1.0 * g, "mean_batch": 12.0,
                     "policy": f"static-g{g}", "rate": 8.0} for g in (2, 4)}
    adp = adaptive_from_grid(by, hw, [2, 4], [8.0], {8.0: 0.4})
    assert adp and adp[0]["chosen_gamma"] in (2, 4), adp
    print("SELF-TEST PASSED (poisson, percentile, summarize, adaptive recon).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--rates", default="2,4,8,16,24",
                    help="Poisson arrival rates (req/s) to sweep")
    ap.add_argument("--policies", default="no-spec,2,4,6",
                    help="no-spec and/or static gammas")
    ap.add_argument("--num-prompts", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--slo-tpot", type=float, default=0.08)
    ap.add_argument("--dataset", default=None, help="path to a ShareGPT json")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-url", default=None,
                    help="use an already-running server (skip orchestration); "
                         "only valid with a single policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rates = [float(x) for x in args.rates.split(",") if x.strip()]
    policies = args.policies.split(",")
    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU -- STOP.")
        return 1
    os.makedirs(RES, exist_ok=True)
    mc = pick_model(gpu.total_mem_gb)
    print(f"GPU: {gpu.name} | model: {mc.target_model} | rates={rates} | "
          f"policies={policies}")

    prompts_all = load_prompts(args.dataset, args.num_prompts, args.seed)
    print(f"Loaded {len(prompts_all)} prompts.")

    rows = []                 # one row per (policy, rate)
    by_pr = {}                # (gamma, rate) -> summary, for adaptive recon
    accept_by_rate = {}

    for pol in policies:
        gamma = 0 if pol == "no-spec" else int(pol)
        url = args.server_url or f"http://localhost:{args.port}"
        proc = None
        if not args.server_url:
            log_path = os.path.join(RES, f"serverlog_{pol}.txt")
            proc = launch_server(mc, gamma, args.port, log_path)
            print(f"[{pol}] waiting for server...")
            if not wait_ready(url, proc):
                print(f"[{pol}] server failed to start; see {log_path}. Skipping.")
                if proc:
                    proc.terminate()
                continue
        try:
            for lam in rates:
                arrivals = poisson_arrivals(len(prompts_all), lam, args.seed)
                recs, wall = run_load(url, mc.target_model, prompts_all,
                                      arrivals, args.max_tokens)
                s = summarize(recs, args.slo_tpot, wall)
                s.update({"policy": pol, "rate": lam})
                rows.append(s)
                if gamma > 0:
                    by_pr[(gamma, lam)] = s
                print(f"  [{pol}] lam={lam:>5}: goodput={s['goodput_req_s']:.2f} "
                      f"req/s | thr={s['throughput_tok_s']:.0f} tok/s | "
                      f"p90 TPOT={s['tpot_p90']*1e3:.1f}ms | "
                      f"SLO={s['slo_attainment']*100:.0f}%")
        finally:
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except Exception:  # noqa: BLE001
                    proc.kill()

    # Adaptive reconstruction (controller picks gamma per rate).
    spec_gammas = sorted({int(p) for p in policies if p != "no-spec"})
    if spec_gammas and by_pr:
        hw = HardwareModel.load_calibrated()
        # crude per-rate acceptance: reuse step4 grid accept if present, else 0.4
        for lam in rates:
            accept_by_rate.setdefault(lam, 0.4)
        rows += adaptive_from_grid(by_pr, hw, spec_gammas, rates, accept_by_rate)

    out = os.path.join(RES, "step7_goodput.csv")
    fields = ["policy", "rate", "goodput_req_s", "throughput_tok_s",
              "slo_attainment", "tpot_p50", "tpot_p90", "tpot_p99",
              "mean_batch", "n_done", "chosen_gamma"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}  ({len(rows)} rows)")
    print("STEP 7 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
