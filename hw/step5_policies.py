"""
step5_policies.py
-----------------
Step 5 (analysis half, CPU-only): turn the measured (gamma x batch) grid from
step4_grid.py into the policy comparison that is the point of the project:

    no-spec, static-gamma (2/4/6), adaptive, and oracle,

on REAL measured throughput and TPOT across the batch-size (load) sweep.

How each policy is realized from the grid:
  * no-spec / static-g : the measured row at that fixed gamma.
  * oracle             : at each batch, the gamma with the highest MEASURED
                         throughput (empirical best).
  * adaptive           : at each batch, the gamma chosen by the SAME
                         controller.AdaptiveController objective (expected
                         accepted tokens per unit GPU time under the CALIBRATED
                         cost model), fed the acceptance MEASURED at that batch.
                         We then report the real measured throughput at the
                         gamma it picked. This is exactly the controller the
                         simulator uses -- here it drives a hardware schedule.

The headline check: does the cost-model-driven adaptive choice track the
empirical oracle, and does it beat every static gamma across the load sweep?

This script touches no GPU; it only reads results_hw/step4_grid.csv and
results_hw/calibrated_constants.json. Run it anywhere the repo + numpy +
matplotlib are available (including a laptop):

    python hw/step5_policies.py --slo-tpot 0.08
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost_model import HardwareModel, expected_tokens_per_step  # noqa: E402
# adaptive_gamma() below applies the exact AdaptiveController objective
# (expected accepted tokens per unit GPU time) from controller.py, restricted
# to the gammas we measured; see that file for the canonical implementation.

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_hw")
GRID_CSV = os.path.join(RES, "step4_grid.csv")


def load_grid(path: str):
    """Return (grid, batches, gammas) where grid[(gamma,batch)] = row dict."""
    grid = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            g, b = int(r["gamma"]), int(r["batch"])
            grid[(g, b)] = {
                "throughput": float(r["throughput_tok_s"]),
                "tpot": float(r["tpot_s"]),
                "accept_fraction": float(r["accept_fraction"]),
                "accept_length": float(r["accept_length"]),
            }
    batches = sorted({b for _, b in grid})
    gammas = sorted({g for g, _ in grid})
    return grid, batches, gammas


def adaptive_gamma(hw: HardwareModel, spec_gammas, batch, alpha_hat):
    """Controller's choice restricted to the gammas we actually measured."""
    best_g, best = spec_gammas[0], -1.0
    for g in spec_gammas:
        score = expected_tokens_per_step(alpha_hat, g) / hw.macro_step_seconds(batch, g)
        if score > best:
            best, best_g = score, g
    return best_g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slo-tpot", type=float, default=0.08,
                    help="per-output-token latency SLO (s) for goodput")
    ap.add_argument("--grid", default=GRID_CSV)
    args = ap.parse_args()

    if not os.path.exists(args.grid):
        print(f"Missing {args.grid}. Run step4_grid.py for each gamma first.")
        return 1

    grid, batches, gammas = load_grid(args.grid)
    spec_gammas = [g for g in gammas if g > 0]
    static_show = [g for g in (2, 4, 6) if g in spec_gammas]
    hw = HardwareModel.load_calibrated()   # uses calibrated_constants.json
    print(f"Loaded grid: gammas={gammas}, batches={batches}")
    print(f"Cost model: t_mem_target={hw.t_mem_target*1e3:.2f}ms  "
          f"t_flop_target={hw.t_flop_target*1e3:.4f}ms/tok (calibrated)")

    # Build per-policy throughput/tpot curves over the batch sweep.
    policies = {}
    if 0 in gammas:
        policies["no-spec"] = [(b, grid[(0, b)]["throughput"], grid[(0, b)]["tpot"])
                               for b in batches if (0, b) in grid]
    for g in static_show:
        policies[f"static-g{g}"] = [(b, grid[(g, b)]["throughput"], grid[(g, b)]["tpot"])
                                    for b in batches if (g, b) in grid]

    # Oracle + adaptive (choose a gamma per batch).
    #   adaptive      : faithful to controller.py -- chooses gamma >= 1.
    #   adaptive+off  : SAME objective, but the candidate set also includes
    #                   gamma=0 (speculation off). The cost model is already
    #                   well-defined there (expected_tokens=1, macro-step=plain
    #                   decode), so this needs NO change to controller.py; it
    #                   just lets the goodput-aware controller stop speculating
    #                   when the batch makes it unprofitable.
    #   oracle        : empirical best measured throughput over ALL gammas
    #                   (including 0), i.e. the achievable envelope.
    oracle, adaptive, adaptive_off, chosen = [], [], [], []
    all_g = ([0] if 0 in gammas else []) + spec_gammas
    for b in batches:
        avail = [g for g in spec_gammas if (g, b) in grid]
        avail_all = [g for g in all_g if (g, b) in grid]
        if not avail:
            continue
        og = max(avail_all, key=lambda g: grid[(g, b)]["throughput"])
        oracle.append((b, grid[(og, b)]["throughput"], grid[(og, b)]["tpot"]))
        alpha_hat = float(np.mean([grid[(g, b)]["accept_fraction"] for g in avail]))
        ag = adaptive_gamma(hw, avail, b, alpha_hat)
        adaptive.append((b, grid[(ag, b)]["throughput"], grid[(ag, b)]["tpot"]))
        ago = adaptive_gamma(hw, avail_all, b, alpha_hat)
        adaptive_off.append((b, grid[(ago, b)]["throughput"], grid[(ago, b)]["tpot"]))
        chosen.append((b, ag, ago, og, alpha_hat))
    policies["oracle"] = oracle
    policies["adaptive"] = adaptive
    policies["adaptive+off"] = adaptive_off

    # --- write comparison CSV ---
    comp = os.path.join(RES, "step5_policies.csv")
    with open(comp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "batch", "throughput_tok_s", "tpot_s",
                    "meets_slo", "slo_bounded_throughput"])
        for name, curve in policies.items():
            for b, thr, tpot in curve:
                meets = tpot <= args.slo_tpot
                w.writerow([name, b, round(thr, 2), round(tpot, 6),
                            int(meets), round(thr if meets else 0.0, 2)])

    # --- write the adaptive-vs-oracle gamma schedule ---
    sched = os.path.join(RES, "step5_gamma_schedule.csv")
    with open(sched, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "adaptive_gamma", "adaptive_off_gamma",
                    "oracle_gamma", "measured_alpha"])
        for b, ag, ago, og, a in chosen:
            w.writerow([b, ag, ago, og, round(a, 4)])

    # --- plots ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    styles = {"no-spec": "--", "oracle": ":", "adaptive": "-"}
    for name, curve in policies.items():
        if not curve:
            continue
        bs = [c[0] for c in curve]
        thr = [c[1] for c in curve]
        lw = 2.6 if name in ("adaptive", "adaptive+off", "oracle") else 1.6
        ax[0].plot(bs, thr, styles.get(name, "-"), marker="o", label=name, lw=lw)
    ax[0].set_xlabel("batch size")
    ax[0].set_ylabel("throughput (tok/s)")
    ax[0].set_title("(a) Measured throughput vs load")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    if chosen:
        bs = [c[0] for c in chosen]
        ax[1].plot(bs, [c[1] for c in chosen], "-o", label="adaptive $\\gamma$", lw=2.4)
        ax[1].plot(bs, [c[2] for c in chosen], "-D", label="adaptive+off $\\gamma$", lw=1.8)
        ax[1].plot(bs, [c[3] for c in chosen], ":s", label="oracle $\\gamma$", lw=2.0)
        ax[1].set_xlabel("batch size")
        ax[1].set_ylabel("chosen $\\gamma$")
        ax[1].set_title("(b) Controller $\\gamma$ tracks the empirical optimum")
        ax[1].grid(alpha=0.3)
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(RES, "step5_policies.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    # --- console summary ---
    print("\nPeak throughput by policy (tok/s):")
    for name, curve in policies.items():
        if curve:
            print(f"  {name:>10}: {max(c[1] for c in curve):8.1f}")
    print(f"\nSLO-bounded throughput (TPOT <= {args.slo_tpot}s), tok/s:")
    for name, curve in policies.items():
        ok = [c[1] for c in curve if c[2] <= args.slo_tpot]
        print(f"  {name:>10}: {max(ok) if ok else 0.0:8.1f}")
    print("\nController gamma vs oracle gamma (adaptive+off may pick 0):")
    for b, ag, ago, og, a in chosen:
        flag = "" if ago == og else "  <-- differs"
        print(f"  B={b:>3}: adaptive={ag}  adaptive+off={ago}  oracle={og}  "
              f"(alpha={a:.2f}){flag}")

    print(f"\nWrote {comp}")
    print(f"Wrote {sched}")
    print(f"Wrote {png}")
    print("STEP 5 DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
