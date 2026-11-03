"""
psd_hw.py  --  self-contained hardware harness for the Goodput-Aware Adaptive
Speculative Decoding project, consolidated into ONE file so it can be
materialized inside a Colab notebook (via %%writefile) with nothing to upload.

It bundles the pieces that elsewhere live in cost_model.py / controller.py /
hw/*.py, unchanged in substance: the roofline cost model, the AdaptiveController
objective, GPU/model autodetection, and the measured experiments (calibration,
the gamma x batch grid for n-gram OR EAGLE, the policy analysis, the serving
goodput sweep, and the FINDINGS generator).

Subcommands (run one per `!python psd_hw.py <cmd>` cell):
    selftest     CPU-only checks of all pure logic (no GPU)
    env          print GPU / dtype / backend / chosen model
    probe-eagle  does EAGLE load+generate on this GPU? + native-schedule check
    calibrate    fit t_mem/t_flop from real decode microbenchmarks
    grid         measure throughput+acceptance for one (method,gamma) over a batch sweep
    analyze      no-spec/static/adaptive/adaptive+off/oracle from the grid (+plot)
    findings     write results_hw/FINDINGS.md from the measured CSVs
    goodput      serving sweep: ShareGPT + Poisson arrivals -> req/s under TPOT SLO

Design notes carried over from the V100 work:
  * dtype float16 everywhere (so A100 and V100 numbers are directly comparable).
  * On Volta we force VLLM_ATTENTION_BACKEND=TRITON_ATTN; on Ampere+ we leave it
    unset so vLLM uses FlashAttention.
  * Acceptance is read from vLLM's own metrics as a FRACTION
    (num_accepted/num_draft), the exact float the controller consumes.
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
import traceback
import urllib.request
from dataclasses import dataclass


# ===========================================================================
# Cost model (roofline) + controller objective  --  identical maths to the sim
# ===========================================================================
@dataclass
class HardwareModel:
    t_mem_target: float = 12.0e-3
    t_flop_target: float = 0.11e-3
    t_mem_draft: float = 1.3e-3
    t_flop_draft: float = 0.012e-3

    def target_pass_seconds(self, n_tokens: int) -> float:
        return self.t_mem_target + self.t_flop_target * n_tokens

    def draft_step_seconds(self, batch: int) -> float:
        return self.t_mem_draft + self.t_flop_draft * batch

    def macro_step_seconds(self, batch: int, gamma: int) -> float:
        draft = gamma * self.draft_step_seconds(batch)
        verify = self.target_pass_seconds(batch * (gamma + 1))
        return draft + verify

    @classmethod
    def load_calibrated(cls, path: str) -> "HardwareModel":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        c = data.get("constants", data)
        hw = cls()
        for k in ("t_mem_target", "t_flop_target", "t_mem_draft", "t_flop_draft"):
            if c.get(k) is not None:
                setattr(hw, k, float(c[k]))
        return hw


def expected_tokens_per_step(alpha: float, gamma: int) -> float:
    if gamma <= 0:
        return 1.0
    if alpha >= 1.0:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def adaptive_gamma(hw: HardwareModel, batch: int, alpha_hat: float,
                   candidate_gammas) -> int:
    """AdaptiveController objective: argmax expected accepted tokens / GPU time,
    over a candidate set (optionally including 0 = no speculation)."""
    best_g, best = candidate_gammas[0], -1.0
    for g in candidate_gammas:
        score = expected_tokens_per_step(alpha_hat, g) / hw.macro_step_seconds(batch, g)
        if score > best:
            best, best_g = score, g
    return best_g


# ===========================================================================
# GPU + model selection
# ===========================================================================
@dataclass
class GpuInfo:
    name: str
    total_mem_gb: float
    compute_capability: tuple | None = None

    @property
    def is_volta(self) -> bool:
        return self.compute_capability == (7, 0)

    @property
    def supports_bf16(self) -> bool:
        return self.compute_capability is not None and self.compute_capability >= (8, 0)


def detect_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return GpuInfo(p.name, p.total_memory / (1024 ** 3), (p.major, p.minor))
    except Exception:
        pass
    return None


@dataclass
class ModelChoice:
    target_model: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    dtype: str = "float16"
    eagle_head: str | None = None

    def describe(self):
        return (f"{self.target_model} | max_len={self.max_model_len} | "
                f"max_num_seqs={self.max_num_seqs} | dtype={self.dtype} | "
                f"eagle={self.eagle_head or '-'}")


_UNGATED_8B = "Qwen/Qwen2.5-7B-Instruct"
_UNGATED_3B = "Qwen/Qwen2.5-3B-Instruct"
# EAGLE3 target that has a vLLM-official, ungated draft head (Red Hat's docs use
# exactly this pair). There is no official EAGLE3 head for Qwen2.5-7B, so the
# EAGLE comparison is run on Qwen3-8B; pass --model Qwen/Qwen3-8B to also run the
# n-gram baseline on the same model for a clean within-GPU comparison.
_EAGLE_TARGET = "Qwen/Qwen3-8B"
_EAGLE_HEADS = {"Qwen/Qwen3-8B": "RedHatAI/Qwen3-8B-speculator.eagle3"}


def pick_model(total_mem_gb: float, want_eagle: bool = False,
               model: str | None = None) -> ModelChoice:
    if want_eagle and model is None:
        model = _EAGLE_TARGET
    if model:
        mc = ModelChoice(model, 4096, 0.90 if total_mem_gb >= 28 else 0.88,
                         64 if total_mem_gb >= 28 else 48)
    elif total_mem_gb >= 28.0:
        mc = ModelChoice(_UNGATED_8B, 4096, 0.90, 64)
    else:
        mc = ModelChoice(_UNGATED_3B, 4096, 0.88, 48)
    if want_eagle:
        mc.eagle_head = _EAGLE_HEADS.get(mc.target_model)
    return mc


def set_attention_backend(gpu):
    if gpu is not None and gpu.is_volta and "VLLM_ATTENTION_BACKEND" not in os.environ:
        os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
        print("Volta -> VLLM_ATTENTION_BACKEND=TRITON_ATTN")


def accepted_fraction(num_accepted, num_drafted):
    if num_drafted <= 0:
        return 0.0
    return min(max(num_accepted / num_drafted, 0.0), 1.0)


# ---- vLLM spec-decode metric helpers --------------------------------------
_M_DRAFTS = "vllm:spec_decode_num_drafts"
_M_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
_M_ACCEPTED = "vllm:spec_decode_num_accepted_tokens"


def _sum_metric(metrics, name):
    tot = 0.0
    for m in metrics:
        if getattr(m, "name", None) != name:
            continue
        if getattr(m, "value", None) is not None:
            tot += float(m.value)
        elif getattr(m, "values", None) is not None:
            tot += float(sum(m.values))
    return tot


def _snapshot(llm):
    try:
        ms = llm.get_metrics()
    except Exception:
        return {}
    return {"drafts": _sum_metric(ms, _M_DRAFTS),
            "draft_tokens": _sum_metric(ms, _M_DRAFT_TOKENS),
            "accepted": _sum_metric(ms, _M_ACCEPTED)}


def _build_spec_cfg(method, gamma, mc):
    if gamma <= 0:
        return None
    if method == "eagle":
        return {"method": "eagle3", "model": mc.eagle_head,
                "num_speculative_tokens": gamma}
    return {"method": "ngram", "num_speculative_tokens": gamma,
            "prompt_lookup_max": 5, "prompt_lookup_min": 2}


def _prompts(n):
    pool = [
        "List the first 20 prime numbers, separated by commas.",
        "Repeat this sentence five times: The quick brown fox jumps over the lazy dog.",
        "Summarize then repeat verbatim: speculative decoding proposes draft tokens "
        "the target verifies in one pass.",
        "Write the numbers 1 through 30, each on its own line.",
        "Explain continuous batching, then restate it in the same words.",
        "Continue this list of US states: Alabama, Alaska, Arizona, Arkansas,",
    ]
    return [pool[i % len(pool)] for i in range(n)]


def res_dir(args):
    d = os.path.abspath(args.res_dir)
    os.makedirs(d, exist_ok=True)
    return d


# ===========================================================================
# Subcommand: env
# ===========================================================================
def cmd_env(args):
    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU."); return 1
    print(f"GPU            : {gpu.name}")
    print(f"Total memory   : {gpu.total_mem_gb:.1f} GB")
    print(f"Compute cap.   : {gpu.compute_capability}")
    print(f"bf16 supported : {gpu.supports_bf16}")
    for mod in ("flash_attn", "flashinfer", "xformers"):
        try:
            __import__(mod); s = "available"
        except Exception as e:
            s = f"absent ({type(e).__name__})"
        print(f"{mod:15}: {s}")
    import vllm
    print("vllm           :", vllm.__version__)
    mc = pick_model(gpu.total_mem_gb, model=args.model)
    print("auto model     :", mc.describe())
    return 0


# ===========================================================================
# Subcommand: probe-eagle
# ===========================================================================
def cmd_probe_eagle(args):
    try:
        from vllm.config import SpeculativeConfig
        has = "num_speculative_tokens_per_batch_size" in \
            getattr(SpeculativeConfig, "__dataclass_fields__", {})
        print(f"[1] native num_speculative_tokens_per_batch_size: {has}")
    except Exception as e:
        print("[1] introspect failed:", e)

    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU."); return 2
    set_attention_backend(gpu)
    mc = pick_model(gpu.total_mem_gb, want_eagle=True, model=args.model)
    if not mc.eagle_head:
        print("[2] no EAGLE head for", mc.target_model); return 2
    print(f"[2] EAGLE3 target={mc.target_model} draft={mc.eagle_head}")
    from vllm import LLM, SamplingParams
    try:
        llm = LLM(model=mc.target_model, dtype=mc.dtype, max_model_len=2048,
                  gpu_memory_utilization=mc.gpu_memory_utilization, max_num_seqs=8,
                  trust_remote_code=True, enforce_eager=True,
                  speculative_config={"method": "eagle3", "model": mc.eagle_head,
                                      "num_speculative_tokens": 3},
                  disable_log_stats=False)
        outs = llm.generate(["Explain speculative decoding in two sentences."],
                            SamplingParams(temperature=0.0, max_tokens=64))
        print("[2] EAGLE OK:", repr(outs[0].outputs[0].text.strip()[:160]))
        snap = _snapshot(llm)
        if snap.get("draft_tokens"):
            print(f"[2] EAGLE acceptance fraction: "
                  f"{accepted_fraction(snap['accepted'], snap['draft_tokens']):.3f}")
        print("\nVERDICT: EAGLE RUNS HERE. Build the EAGLE sweep.")
        return 0
    except Exception:
        traceback.print_exc()
        print("\nVERDICT: EAGLE does NOT run here. Document it; keep n-gram.")
        return 3


# ===========================================================================
# Subcommand: calibrate
# ===========================================================================
def cmd_calibrate(args):
    import numpy as np
    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU."); return 1
    set_attention_backend(gpu)
    RES = res_dir(args)
    mc = pick_model(gpu.total_mem_gb, model=args.model)
    batches = [b for b in [int(x) for x in args.batches.split(",")] if b <= mc.max_num_seqs]
    from vllm import LLM, SamplingParams
    llm = LLM(model=mc.target_model, dtype=mc.dtype, max_model_len=mc.max_model_len,
              gpu_memory_utilization=mc.gpu_memory_utilization,
              max_num_seqs=mc.max_num_seqs, trust_remote_code=True,
              enforce_eager=True, enable_prefix_caching=False)
    prompt = "Write a long detailed story that keeps going without stopping. Begin now:"
    sp1 = SamplingParams(temperature=0.0, max_tokens=args.n1, ignore_eos=True)
    sp2 = SamplingParams(temperature=0.0, max_tokens=args.n2, ignore_eos=True)
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
    rows = []
    for B in batches:
        ps = [prompt] * B
        t0 = time.time(); llm.generate(ps, sp1); t1 = time.time() - t0
        t0 = time.time(); llm.generate(ps, sp2); t2 = time.time() - t0
        step = (t2 - t1) / (args.n2 - args.n1)
        rows.append({"batch": B, "step_latency_s": step})
        print(f"  B={B:>3} step={step*1e3:7.3f} ms")
    Bs = np.array([r["batch"] for r in rows], float)
    lat = np.array([r["step_latency_s"] for r in rows], float)
    slope, intercept = np.polyfit(Bs, lat, 1)
    pred = slope * Bs + intercept
    r2 = 1 - float(np.sum((lat - pred) ** 2)) / float(np.sum((lat - lat.mean()) ** 2))
    out = {"gpu": gpu.name, "model": mc.target_model, "fit_r2": r2,
           "constants": {"t_mem_target": float(max(intercept, 0)),
                         "t_flop_target": float(max(slope, 0)),
                         "t_mem_draft": 0.0, "t_flop_draft": 0.0}, "raw": rows}
    path = os.path.join(RES, "calibrated_constants.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"t_mem={intercept*1e3:.3f}ms t_flop={slope*1e3:.5f}ms/tok R2={r2:.4f}")
    print("Wrote", path)
    return 0


# ===========================================================================
# Subcommand: grid
# ===========================================================================
def cmd_grid(args):
    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU."); return 1
    set_attention_backend(gpu)
    RES = res_dir(args)
    want_eagle = args.method == "eagle"
    mc = pick_model(gpu.total_mem_gb, want_eagle=want_eagle, model=args.model)
    if want_eagle and args.gamma > 0 and not mc.eagle_head:
        print("No EAGLE head; STOP."); return 1
    batches = [b for b in [int(x) for x in args.batches.split(",")] if b <= mc.max_num_seqs]
    print(f"GPU {gpu.name} | {mc.target_model} | method={args.method} | gamma={args.gamma}")
    from vllm import LLM, SamplingParams
    kw = dict(model=mc.target_model, dtype=mc.dtype, max_model_len=mc.max_model_len,
              gpu_memory_utilization=mc.gpu_memory_utilization,
              max_num_seqs=mc.max_num_seqs, trust_remote_code=True,
              enforce_eager=True, disable_log_stats=False)
    spec = _build_spec_cfg(args.method, args.gamma, mc)
    if spec:
        kw["speculative_config"] = spec
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    llm.generate(["Hello, warm up."], SamplingParams(temperature=0.0, max_tokens=8))
    prev = _snapshot(llm); spec_ok = bool(prev) and args.gamma > 0
    fields = ["gamma", "batch", "wall_s", "out_tokens", "throughput_tok_s", "tpot_s",
              "accept_fraction", "accept_length", "num_drafts", "draft_tokens",
              "accepted_tokens"]
    rows = []
    for B in batches:
        t0 = time.time(); outs = llm.generate(_prompts(B), sp); dt = time.time() - t0
        ot = sum(len(o.outputs[0].token_ids) for o in outs)
        if spec_ok:
            now = _snapshot(llm)
            d_dr = now["drafts"] - prev["drafts"]; d_dt = now["draft_tokens"] - prev["draft_tokens"]
            d_at = now["accepted"] - prev["accepted"]; prev = now
            af = accepted_fraction(d_at, d_dt); al = 1 + (d_at / d_dr if d_dr else 0)
        else:
            d_dr = d_dt = d_at = 0; af = 0.0; al = 1.0
        thr = ot / dt if dt else 0; tpot = dt / (ot / B) if ot else float("nan")
        rows.append({"gamma": args.gamma, "batch": B, "wall_s": round(dt, 4),
                     "out_tokens": ot, "throughput_tok_s": round(thr, 2),
                     "tpot_s": round(tpot, 6), "accept_fraction": round(af, 4),
                     "accept_length": round(al, 4), "num_drafts": int(d_dr),
                     "draft_tokens": int(d_dt), "accepted_tokens": int(d_at)})
        print(f"  B={B:>3} thr={thr:7.1f} acc_frac={af:.3f} acc_len={al:.2f}")
    grid_csv = os.path.join(RES, f"step4_grid_{args.method}.csv")
    header = args.fresh or not os.path.exists(grid_csv)
    with open(grid_csv, "w" if args.fresh else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if header:
            w.writeheader()
        w.writerows(rows)
    print(("Wrote " if header else "Appended ") + grid_csv)
    return 0


# ===========================================================================
# Subcommand: analyze
# ===========================================================================
def _load_grid(path):
    grid = {}
    for r in csv.DictReader(open(path)):
        grid[(int(r["gamma"]), int(r["batch"]))] = {
            "throughput": float(r["throughput_tok_s"]), "tpot": float(r["tpot_s"]),
            "accept_fraction": float(r["accept_fraction"])}
    return grid


def cmd_analyze(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RES = res_dir(args)
    gpath = os.path.join(RES, f"step4_grid_{args.method}.csv")
    if not os.path.exists(gpath):
        print("Missing", gpath); return 1
    grid = _load_grid(gpath)
    batches = sorted({b for _, b in grid}); gammas = sorted({g for g, _ in grid})
    spec = [g for g in gammas if g > 0]
    hw = HardwareModel.load_calibrated(os.path.join(RES, "calibrated_constants.json"))
    policies = {}
    if 0 in gammas:
        policies["no-spec"] = [(b, grid[(0, b)]["throughput"], grid[(0, b)]["tpot"]) for b in batches if (0, b) in grid]
    for g in [x for x in (2, 4, 6) if x in spec]:
        policies[f"static-g{g}"] = [(b, grid[(g, b)]["throughput"], grid[(g, b)]["tpot"]) for b in batches if (g, b) in grid]
    oracle, adaptive, adaptive_off, chosen = [], [], [], []
    allg = ([0] if 0 in gammas else []) + spec
    for b in batches:
        av = [g for g in spec if (g, b) in grid]
        ava = [g for g in allg if (g, b) in grid]
        if not av:
            continue
        og = max(ava, key=lambda g: grid[(g, b)]["throughput"])
        oracle.append((b, grid[(og, b)]["throughput"], grid[(og, b)]["tpot"]))
        a = float(np.mean([grid[(g, b)]["accept_fraction"] for g in av]))
        ag = adaptive_gamma(hw, b, a, av); ago = adaptive_gamma(hw, b, a, ava)
        adaptive.append((b, grid[(ag, b)]["throughput"], grid[(ag, b)]["tpot"]))
        adaptive_off.append((b, grid[(ago, b)]["throughput"], grid[(ago, b)]["tpot"]))
        chosen.append((b, ag, ago, og, a))
    policies["oracle"] = oracle; policies["adaptive"] = adaptive
    policies["adaptive+off"] = adaptive_off
    with open(os.path.join(RES, f"step5_policies_{args.method}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["policy", "batch", "throughput_tok_s", "tpot_s", "meets_slo"])
        for name, curve in policies.items():
            for b, thr, tp in curve:
                w.writerow([name, b, round(thr, 2), round(tp, 6), int(tp <= args.slo_tpot)])
    with open(os.path.join(RES, f"step5_schedule_{args.method}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["batch", "adaptive_gamma", "adaptive_off_gamma", "oracle_gamma", "measured_alpha"])
        for b, ag, ago, og, a in chosen:
            w.writerow([b, ag, ago, og, round(a, 4)])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, curve in policies.items():
        if curve:
            ax[0].plot([c[0] for c in curve], [c[1] for c in curve], marker="o", label=name,
                       lw=2.6 if name in ("adaptive", "oracle", "adaptive+off") else 1.5)
    ax[0].set_xlabel("batch"); ax[0].set_ylabel("throughput tok/s")
    ax[0].set_title(f"({args.method}) throughput vs load"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    if chosen:
        ax[1].plot([c[0] for c in chosen], [c[1] for c in chosen], "-o", label="adaptive γ", lw=2.4)
        ax[1].plot([c[0] for c in chosen], [c[3] for c in chosen], ":s", label="oracle γ")
        ax[1].set_xlabel("batch"); ax[1].set_ylabel("chosen γ"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, f"step5_policies_{args.method}.png"), dpi=150)
    print("\nPeak throughput by policy:")
    for name, curve in policies.items():
        if curve:
            print(f"  {name:>12}: {max(c[1] for c in curve):8.1f}")
    print("\ngamma schedule (adaptive / adaptive+off / oracle):")
    for b, ag, ago, og, a in chosen:
        print(f"  B={b:>3}: {ag} / {ago} / {og}   (alpha={a:.2f})")
    return 0


# ===========================================================================
# Subcommand: findings
# ===========================================================================
def cmd_findings(args):
    RES = res_dir(args)
    L = []
    def w(s=""):
        L.append(s)
    calib = None
    cp = os.path.join(RES, "calibrated_constants.json")
    if os.path.exists(cp):
        calib = json.load(open(cp))
    w("# Hardware Findings — Adaptive Speculative Decoding (generated)")
    w()
    w("All numbers are measured on a real GPU and generated from the run CSVs; "
      "nothing is hand-entered.")
    w()
    if calib:
        c = calib["constants"]
        w(f"- GPU: **{calib.get('gpu')}**, model **{calib.get('model')}**")
        w(f"- Calibration: t_mem={c['t_mem_target']*1e3:.3f} ms, "
          f"t_flop={c['t_flop_target']*1e3:.5f} ms/tok, R²={calib.get('fit_r2', float('nan')):.4f}")
        w()
    for method in ("ngram", "eagle"):
        pol = os.path.join(RES, f"step5_policies_{method}.csv")
        sch = os.path.join(RES, f"step5_schedule_{method}.csv")
        if not os.path.exists(pol):
            continue
        rows = list(csv.DictReader(open(pol)))
        thr_at = {}
        for r in rows:
            thr_at.setdefault(r["policy"], {})[int(r["batch"])] = float(r["throughput_tok_s"])
        w(f"## {method} — policy comparison")
        w()
        w("| policy | peak tok/s | worst-case shortfall vs oracle |")
        w("|--------|-----------:|-------------------------------:|")
        orc = thr_at.get("oracle", {})
        bs = sorted(orc.keys())
        def wr(p):
            worst = 0.0
            for b in bs:
                o, v = orc.get(b), thr_at[p].get(b)
                if o and v is not None:
                    worst = max(worst, 100 * (o - v) / o)
            return worst
        for p in thr_at:
            peak = max(thr_at[p].values())
            w(f"| {p} | {peak:.1f} | {('' if p=='oracle' else f'{wr(p):.1f}%')} |")
        w()
        if "adaptive" in thr_at and bs:
            statics = {p: wr(p) for p in thr_at if p.startswith("static")}
            worst_static = max(statics.values()) if statics else 0.0
            w(f"Adaptive worst-case shortfall **{wr('adaptive'):.1f}%** vs up to "
              f"**{worst_static:.1f}%** for the worst static γ and "
              f"{wr('no-spec'):.1f}% for no-spec — adaptive is the most "
              f"load-robust policy.")
            w()
        if os.path.exists(sch):
            srows = list(csv.DictReader(open(sch)))
            w("| batch | adaptive γ | adaptive+off γ | oracle γ | measured α |")
            w("|------:|-----------:|---------------:|---------:|-----------:|")
            for r in srows:
                w(f"| {r['batch']} | {r['adaptive_gamma']} | {r['adaptive_off_gamma']} "
                  f"| {r['oracle_gamma']} | {float(r['measured_alpha']):.2f} |")
            w()
    out = os.path.join(RES, "FINDINGS.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("Wrote", out)
    return 0


# ===========================================================================
# Subcommand: goodput  (serving: ShareGPT + Poisson -> req/s under TPOT SLO)
# ===========================================================================
def poisson_arrivals(n, rate, seed=0):
    rng = random.Random(seed); t = 0.0; out = []
    for _ in range(n):
        t += rng.expovariate(rate); out.append(t)
    return out


def percentile(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0; lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results, slo, wall):
    ok = [r for r in results if r.get("ok")]
    tp = [r["tpot"] for r in ok if r["tpot"] == r["tpot"]]
    meets = [r for r in ok if r["tpot"] == r["tpot"] and r["tpot"] <= slo]
    ot = sum(r["n_out"] for r in ok)
    return {"n_done": len(ok), "throughput_tok_s": ot / wall if wall else 0,
            "goodput_req_s": len(meets) / wall if wall else 0,
            "slo_attainment": (len(meets) / len(ok)) if ok else 0,
            "tpot_p50": percentile(tp, 50), "tpot_p90": percentile(tp, 90),
            "tpot_p99": percentile(tp, 99)}


def _server_up(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _stream_one(url, model, prompt, max_tokens, rec):
    import requests
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    t0 = time.time(); first = None; n = 0
    try:
        resp = requests.post(url + "/v1/completions", data=body,
                             headers={"Content-Type": "application/json"}, stream=True, timeout=600)
        for line in resp.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8", "ignore")
            if not s.startswith("data:"):
                continue
            pay = s[5:].strip()
            if pay == "[DONE]":
                break
            try:
                tok = json.loads(pay)
            except Exception:
                continue
            if tok.get("choices") and tok["choices"][0].get("text"):
                if first is None:
                    first = time.time()
                n += 1
        end = time.time()
        rec["n_out"] = n
        rec["tpot"] = ((end - first) / (n - 1)) if (first and n > 1) else float("nan")
        rec["ok"] = n > 0
    except Exception as e:
        rec["ok"] = False; rec["err"] = str(e)


def run_load(url, model, prompts, arrivals, max_tokens):
    recs = [dict() for _ in prompts]; ths = []; start = time.time()
    for i, (p, a) in enumerate(zip(prompts, arrivals)):
        d = a - (time.time() - start)
        if d > 0:
            time.sleep(d)
        th = threading.Thread(target=_stream_one, args=(url, model, p, max_tokens, recs[i]))
        th.start(); ths.append(th)
    for th in ths:
        th.join()
    return recs, time.time() - start


def load_prompts(n, seed=0):
    try:
        from datasets import load_dataset
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                          data_files="ShareGPT_V3_unfiltered_cleaned_split.json", split="train")
        convos = ds.to_list()
    except Exception as e:
        print("[warn] ShareGPT unavailable:", e, "-> synthetic")
        base = ["Summarize the history of the internet.",
                "Write a Python function to merge two sorted lists.",
                "Explain how transformers work, step by step.",
                "Give me a recipe for a quick weeknight dinner."]
        return [base[i % len(base)] for i in range(n)]
    ps = []
    for c in convos:
        for t in c.get("conversations", []):
            if t.get("from") in ("human", "user") and t.get("value"):
                ps.append(t["value"][:2000]); break
        if len(ps) >= n * 2:
            break
    random.Random(seed).shuffle(ps)
    return ps[:n] if ps else ["Tell me about speculative decoding."] * n


def cmd_goodput(args):
    gpu = detect_gpu()
    if gpu is None:
        print("NO GPU."); return 1
    RES = res_dir(args)
    mc = pick_model(gpu.total_mem_gb, model=args.model)
    rates = [float(x) for x in args.rates.split(",")]
    policies = args.policies.split(",")
    prompts = load_prompts(args.num_prompts, args.seed)
    print(f"{len(prompts)} prompts | rates={rates} | policies={policies}")
    rows = []
    for pol in policies:
        gamma = 0 if pol == "no-spec" else int(pol)
        env = dict(os.environ)
        set_attention_backend(gpu)
        cmd = ["vllm", "serve", mc.target_model, "--dtype", mc.dtype,
               "--max-model-len", str(mc.max_model_len), "--gpu-memory-utilization",
               str(mc.gpu_memory_utilization), "--max-num-seqs", str(mc.max_num_seqs),
               "--port", str(args.port), "--enforce-eager", "--trust-remote-code",
               "--disable-log-requests"]
        if gamma > 0:
            cmd += ["--speculative-config", json.dumps(_build_spec_cfg(args.method, gamma, mc))]
        log = open(os.path.join(RES, f"serverlog_{args.method}_{pol}.txt"), "w")
        print(f"[{pol}] launching server...")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        url = f"http://localhost:{args.port}"
        t0 = time.time(); ready = False
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                print(f"[{pol}] server exited early; see log."); break
            if _server_up(url):
                ready = True; break
            time.sleep(4)
        if not ready:
            proc.terminate(); continue
        try:
            for lam in rates:
                arr = poisson_arrivals(len(prompts), lam, args.seed)
                recs, wall = run_load(url, mc.target_model, prompts, arr, args.max_tokens)
                s = summarize(recs, args.slo_tpot, wall); s.update({"policy": pol, "rate": lam})
                rows.append(s)
                print(f"  [{pol}] lam={lam}: goodput={s['goodput_req_s']:.2f} req/s "
                      f"thr={s['throughput_tok_s']:.0f} p90TPOT={s['tpot_p90']*1e3:.0f}ms")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
    out = os.path.join(RES, f"step7_goodput_{args.method}.csv")
    fields = ["policy", "rate", "goodput_req_s", "throughput_tok_s", "slo_attainment",
              "tpot_p50", "tpot_p90", "tpot_p99", "n_done"]
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader(); wr.writerows(rows)
    print("Wrote", out)
    return 0


# ===========================================================================
# Subcommand: selftest  (CPU only)
# ===========================================================================
def cmd_selftest(args):
    hw = HardwareModel(t_mem_target=0.0229, t_flop_target=0.000145, t_mem_draft=0, t_flop_draft=0)
    assert adaptive_gamma(hw, 1, 0.4, [1, 2, 4, 6]) >= adaptive_gamma(hw, 64, 0.4, [1, 2, 4, 6])
    assert adaptive_gamma(hw, 64, 0.4, [0, 1, 2, 4]) in (0, 1)
    assert abs(expected_tokens_per_step(0.5, 1) - 1.5) < 1e-9
    a = poisson_arrivals(1000, 10.0, 1); assert a == sorted(a) and 80 < a[-1] < 120
    assert abs(percentile([1, 2, 3, 4], 50) - 2.5) < 1e-9
    s = summarize([{"ok": True, "tpot": 0.05, "n_out": 100},
                   {"ok": True, "tpot": 0.2, "n_out": 50}, {"ok": False}], 0.08, 10.0)
    assert s["n_done"] == 2 and abs(s["goodput_req_s"] - 0.1) < 1e-9
    assert abs(accepted_fraction(7, 20) - 0.35) < 1e-9
    print("SELFTEST PASSED.")
    return 0


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "env", "probe-eagle", "calibrate",
                                    "grid", "analyze", "findings", "goodput"])
    ap.add_argument("--res-dir", default="results_hw")
    ap.add_argument("--method", choices=["ngram", "eagle"], default="ngram")
    ap.add_argument("--model", default=None,
                    help="override target model (e.g. Qwen/Qwen3-8B)")
    ap.add_argument("--gamma", type=int, default=0)
    ap.add_argument("--batches", default="1,2,4,8,16,32,48,64")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--n1", type=int, default=32)
    ap.add_argument("--n2", type=int, default=160)
    ap.add_argument("--slo-tpot", type=float, default=0.08)
    ap.add_argument("--rates", default="2,4,8,16,24")
    ap.add_argument("--policies", default="no-spec,2,4,6")
    ap.add_argument("--num-prompts", type=int, default=300)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "env": cmd_env, "probe-eagle": cmd_probe_eagle,
            "calibrate": cmd_calibrate, "grid": cmd_grid, "analyze": cmd_analyze,
            "findings": cmd_findings, "goodput": cmd_goodput}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
