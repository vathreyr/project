"""
run_experiments.py
------------------
Reproduces the three preliminary experiments reported in the mid-term:

  E1. Load-vs-best-gamma characterization: for a sweep of fixed batch sizes,
      which static gamma maximizes tokens/sec? Shows the optimum moving with
      load -- the motivation for the whole project.
  E2. Policy comparison across a load sweep: no-spec, static gammas, the
      adaptive controller, and the oracle, on throughput and goodput vs
      arrival rate.
  E3. Acceptance drift: a workload where acceptance decays within a
      generation, showing the controller tracking it.

Outputs: CSVs + PNGs under results/. All numbers are simulator outputs; the
timing constants are illustrative (see cost_model.py).

Run:  ../.venv/bin/python run_experiments.py
"""

from __future__ import annotations
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cost_model import HardwareModel, expected_tokens_per_step
from controller import (StaticPolicy, NoSpecPolicy, AdaptiveController,
                        OraclePolicy)
from simulator import WorkloadConfig, simulate

HW = HardwareModel()
GAMMA_MAX = 8
RES = "results"


# ----------------------------------------------------------------------------
# E1: load-vs-best-gamma characterization (closed form from the cost model)
# ----------------------------------------------------------------------------
def e1_characterization(alpha=0.72):
    batches = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    rows = []
    best_curve = []
    for B in batches:
        best_g, best_tps = 1, -1.0
        per_g = {}
        for g in range(1, GAMMA_MAX + 1):
            toks = expected_tokens_per_step(alpha, g)
            tps = toks / HW.macro_step_seconds(B, g)   # tokens/s per request
            per_g[g] = tps
            if tps > best_tps:
                best_tps, best_g = tps, g
        best_curve.append((B, best_g))
        rows.append({"batch": B, "best_gamma": best_g,
                     **{f"tps_g{g}": round(per_g[g], 2) for g in per_g}})

    with open(f"{RES}/e1_characterization.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Heatmap-ish line plot: tokens/s vs gamma for a few batch sizes.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for B in [1, 8, 32, 64, 128]:
        ys = [expected_tokens_per_step(alpha, g) / HW.macro_step_seconds(B, g)
              for g in range(1, GAMMA_MAX + 1)]
        ax[0].plot(range(1, GAMMA_MAX + 1), ys, marker="o", label=f"B={B}")
    ax[0].set_xlabel("speculation length $\\gamma$")
    ax[0].set_ylabel("tokens/s per request")
    ax[0].set_title("(a) Throughput vs $\\gamma$ at fixed load")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    Bs = [b for b, _ in best_curve]
    Gs = [g for _, g in best_curve]
    ax[1].plot(Bs, Gs, marker="s", color="crimson")
    ax[1].set_xlabel("batch size $B$")
    ax[1].set_ylabel("best static $\\gamma$")
    ax[1].set_title("(b) Optimal $\\gamma$ shrinks as load grows")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RES}/e1_characterization.png", dpi=150)
    plt.close(fig)
    print("E1 best (B -> gamma):", best_curve)
    return best_curve


# ----------------------------------------------------------------------------
# E2: policy comparison across a load sweep
# ----------------------------------------------------------------------------
def e2_load_sweep():
    lambdas = [5, 10, 20, 35, 50, 70, 90, 120]
    policies = {
        "no-spec": lambda: NoSpecPolicy(),
        "static-g2": lambda: StaticPolicy(2),
        "static-g5": lambda: StaticPolicy(5),
        "adaptive": lambda: AdaptiveController(HW, gamma_max=GAMMA_MAX),
        "oracle": lambda: OraclePolicy(HW, gamma_max=GAMMA_MAX),
    }
    results = {name: [] for name in policies}
    for lam in lambdas:
        cfg = WorkloadConfig(n_requests=1500, lam=lam, seed=7)
        for name, make in policies.items():
            r = simulate(make(), cfg, HW, max_batch=64, slo_tpot=0.04)
            results[name].append(r)

    # CSV
    with open(f"{RES}/e2_load_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "lambda", "throughput_tps", "goodput_rps",
                    "slo_attain", "tpot_p50", "tpot_p90", "tpot_p99",
                    "mean_batch", "mean_gamma"])
        for name in policies:
            for r in results[name]:
                w.writerow([name, r.lam, round(r.throughput, 1),
                            round(r.goodput, 3), round(r.slo_attainment, 3),
                            round(r.tpot_p50, 4), round(r.tpot_p90, 4),
                            round(r.tpot_p99, 4), round(r.mean_batch, 1),
                            round(r.mean_gamma, 2)])

    # Plots: throughput and goodput vs lambda
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    styles = {"no-spec": "--", "static-g2": "-", "static-g5": "-",
              "adaptive": "-", "oracle": ":"}
    for name in policies:
        lam = [r.lam for r in results[name]]
        tp = [r.throughput for r in results[name]]
        gp = [r.goodput for r in results[name]]
        lw = 2.4 if name in ("adaptive", "oracle") else 1.5
        ax[0].plot(lam, tp, styles[name], marker="o", label=name, lw=lw)
        ax[1].plot(lam, gp, styles[name], marker="o", label=name, lw=lw)
    ax[0].set_xlabel("arrival rate $\\lambda$ (req/s)")
    ax[0].set_ylabel("throughput (tok/s)")
    ax[0].set_title("(a) Throughput vs load")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("arrival rate $\\lambda$ (req/s)")
    ax[1].set_ylabel("goodput (req/s meeting SLO)")
    ax[1].set_title("(b) Goodput vs load")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RES}/e2_load_sweep.png", dpi=150)
    plt.close(fig)

    # Print a compact summary table
    print("\nE2 goodput (req/s):")
    print("lambda   " + "  ".join(f"{n:>10}" for n in policies))
    for i, lam in enumerate(lambdas):
        print(f"{lam:>6}   " + "  ".join(
            f"{results[n][i].goodput:>10.2f}" for n in policies))
    return results, lambdas


# ----------------------------------------------------------------------------
# E3: acceptance drift within a generation
# ----------------------------------------------------------------------------
def e3_drift():
    lam = 30
    out = {}
    for name, make in {
        "static-g5": lambda: StaticPolicy(5),
        "static-g2": lambda: StaticPolicy(2),
        "adaptive": lambda: AdaptiveController(HW, gamma_max=GAMMA_MAX),
    }.items():
        cfg = WorkloadConfig(n_requests=1500, lam=lam, drift=0.0012, seed=11)
        out[name] = simulate(make(), cfg, HW, max_batch=64, slo_tpot=0.04)

    with open(f"{RES}/e3_drift.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "throughput_tps", "goodput_rps", "tpot_p90",
                    "mean_gamma"])
        for name, r in out.items():
            w.writerow([name, round(r.throughput, 1), round(r.goodput, 3),
                        round(r.tpot_p90, 4), round(r.mean_gamma, 2)])
    print("\nE3 drift workload:")
    for name, r in out.items():
        print(f"  {name:>10}: thr={r.throughput:7.1f} tok/s  "
              f"goodput={r.goodput:5.2f} rps  mean_gamma={r.mean_gamma:.2f}")
    return out


if __name__ == "__main__":
    import os
    os.makedirs(RES, exist_ok=True)
    print("=" * 60)
    e1_characterization()
    e2_load_sweep()
    e3_drift()
    print("=" * 60)
    print(f"Wrote CSVs and PNGs to {RES}/")
