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
                        OraclePolicy, BatchAwareFixedAlpha)
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
    ax[1].plot(Bs, Gs, marker="s", ms=6, color="crimson",
               drawstyle="steps-post")
    # Annotate every point with its gamma so the figure cannot be misread and
    # always matches the table, which is generated from the same best_curve.
    for b, g in best_curve:
        ax[1].annotate(str(g), (b, g), textcoords="offset points",
                       xytext=(0, 7), ha="center", fontsize=8, color="crimson")
    ax[1].set_xlabel("batch size $B$")
    ax[1].set_ylabel("best static $\\gamma$")
    ax[1].set_yticks(range(0, GAMMA_MAX + 1))
    ax[1].set_ylim(0, 5)
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
SEEDS = [7, 11, 17, 23, 29]          # repeated runs so differences are not noise
SLO_TPOT = 0.045                     # per-output-token SLO (s/token)


def _mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def e2_load_sweep():
    """Policy comparison across a load sweep, averaged over multiple seeds.

    Reports throughput and goodput where goodput is end-to-end: requests/s whose
    normalized latency INCLUDING queue wait meets the SLO (see simulator). We
    sweep a wide static-gamma set so the claim is anchored to tested baselines.
    """
    lambdas = [5, 10, 20, 35, 50, 70, 90, 120, 150]
    policies = {
        "no-spec": lambda: NoSpecPolicy(),
        "static-g2": lambda: StaticPolicy(2),
        "static-g3": lambda: StaticPolicy(3),
        "static-g4": lambda: StaticPolicy(4),
        "static-g5": lambda: StaticPolicy(5),
        "static-g6": lambda: StaticPolicy(6),
        "adaptive": lambda: AdaptiveController(HW, gamma_max=GAMMA_MAX),
        "oracle": lambda: OraclePolicy(HW, gamma_max=GAMMA_MAX),
    }
    # agg[name][lam] = dict of mean/std for throughput and end-to-end goodput
    agg = {name: {} for name in policies}
    for lam in lambdas:
        for name, make in policies.items():
            thr, gp = [], []
            for sd in SEEDS:
                cfg = WorkloadConfig(n_requests=1500, lam=lam, seed=sd)
                r = simulate(make(), cfg, HW, max_batch=64, slo_tpot=SLO_TPOT)
                thr.append(r.throughput); gp.append(r.goodput_e2e)
            tm, ts = _mean_std(thr); gm, gs = _mean_std(gp)
            agg[name][lam] = dict(thr_mean=tm, thr_std=ts, gp_mean=gm, gp_std=gs)

    with open(f"{RES}/e2_load_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "lambda", "throughput_mean", "throughput_std",
                    "goodput_e2e_mean", "goodput_e2e_std", "n_seeds"])
        for name in policies:
            for lam in lambdas:
                a = agg[name][lam]
                w.writerow([name, lam, round(a["thr_mean"], 1),
                            round(a["thr_std"], 1), round(a["gp_mean"], 3),
                            round(a["gp_std"], 3), len(SEEDS)])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in policies:
        lam = lambdas
        tp = [agg[name][x]["thr_mean"] for x in lam]
        tps = [agg[name][x]["thr_std"] for x in lam]
        gp = [agg[name][x]["gp_mean"] for x in lam]
        gps = [agg[name][x]["gp_std"] for x in lam]
        lw = 2.6 if name in ("adaptive", "oracle") else 1.3
        st = "--" if name == "no-spec" else (":" if name == "oracle" else "-")
        # Error bars show +/- 1 standard deviation over the seeds.
        ax[0].errorbar(lam, tp, yerr=tps, fmt=st, marker="o", ms=4,
                       capsize=2, elinewidth=0.8, label=name, lw=lw)
        ax[1].errorbar(lam, gp, yerr=gps, fmt=st, marker="o", ms=4,
                       capsize=2, elinewidth=0.8, label=name, lw=lw)
    ax[0].set_xlabel("arrival rate $\\lambda$ (req/s)")
    ax[0].set_ylabel("throughput (tok/s)")
    ax[0].set_title("(a) Throughput vs load (mean of %d seeds)" % len(SEEDS))
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("arrival rate $\\lambda$ (req/s)")
    ax[1].set_ylabel("end-to-end goodput (req/s under SLO)")
    ax[1].set_title("(b) Goodput vs load")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{RES}/e2_load_sweep.png", dpi=150)
    plt.close(fig)

    print("\nE2 end-to-end goodput (req/s), mean +/- std over %d seeds:" % len(SEEDS))
    hdr = "lambda  " + " ".join(f"{n:>11}" for n in policies)
    print(hdr)
    for lam in lambdas:
        row = f"{lam:>6}  " + " ".join(
            f"{agg[n][lam]['gp_mean']:>5.1f}" + "±" +
            f"{agg[n][lam]['gp_std']:<4.1f}" for n in policies)
        print(row)
    return agg, lambdas


# ----------------------------------------------------------------------------
# E3: acceptance drift within a generation
# ----------------------------------------------------------------------------
def e3_drift():
    """Within-generation acceptance drift, with an ABLATION that isolates the
    contribution of online acceptance estimation.

    The 'batch-only' policy is batch-aware (it picks gamma from the batch size
    through the same cost-model objective) but uses a FIXED acceptance rate set
    to the undrifted mean (0.72), so it cannot react to the decay. The full
    adaptive policy additionally tracks acceptance online. The gap between the
    two is exactly the value of the online estimate, which the static-gamma
    baselines cannot isolate.
    """
    lam = 30
    alpha0 = 0.72
    makers = {
        "static-g2": lambda: StaticPolicy(2),
        "static-g5": lambda: StaticPolicy(5),
        "batch-only": lambda: BatchAwareFixedAlpha(HW, GAMMA_MAX, alpha0),
        "adaptive": lambda: AdaptiveController(HW, gamma_max=GAMMA_MAX),
    }
    out = {}
    for name, make in makers.items():
        thr, gp, mg = [], [], []
        for sd in SEEDS:
            cfg = WorkloadConfig(n_requests=1500, lam=lam, alpha_mean=alpha0,
                                 drift=0.0012, seed=sd)
            r = simulate(make(), cfg, HW, max_batch=64, slo_tpot=SLO_TPOT)
            thr.append(r.throughput); gp.append(r.goodput_e2e); mg.append(r.mean_gamma)
        out[name] = dict(thr=_mean_std(thr), gp=_mean_std(gp), mg=_mean_std(mg))

    with open(f"{RES}/e3_drift.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "throughput_mean", "throughput_std",
                    "goodput_e2e_mean", "goodput_e2e_std", "mean_gamma_mean"])
        for name, r in out.items():
            w.writerow([name, round(r["thr"][0], 1), round(r["thr"][1], 1),
                        round(r["gp"][0], 3), round(r["gp"][1], 3),
                        round(r["mg"][0], 2)])
    print("\nE3 drift workload (ablation), mean +/- std over %d seeds:" % len(SEEDS))
    for name, r in out.items():
        print(f"  {name:>11}: thr={r['thr'][0]:7.1f}+/-{r['thr'][1]:4.1f}  "
              f"goodput_e2e={r['gp'][0]:5.2f}+/-{r['gp'][1]:4.2f}  "
              f"mean_gamma={r['mg'][0]:.2f}")
    print("  (adaptive minus batch-only isolates the online-acceptance benefit)")
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
