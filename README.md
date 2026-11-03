# Adaptive Speculative Decoding — simulation + hardware

Two parts: (1) a CPU **simulation study** that designs the adaptive controller,
and (2) a **hardware study** (`hw/`) that runs it through real vLLM speculative
decoding on a PSC V100 and reports measured results in `results_hw/FINDINGS.md`.
The hardware path reuses the simulator's controller/cost-model objects
unchanged, so the two stay consistent.

## Files
- `cost_model.py` — roofline-style latency model for a propose+verify step.
  Constants are illustrative, to be calibrated against real vLLM microbenchmarks.
- `controller.py` — the policies under study: static-γ, no-spec, the
  **AdaptiveController** (EWMA acceptance + cost-model-driven γ choice), and an
  oracle upper bound.
- `simulator.py` — discrete-event server: Poisson arrivals, continuous
  batching, heterogeneous/drifting acceptance, TPOT/goodput measurement.
- `run_experiments.py` — reproduces E1 (load-vs-best-γ), E2 (policy sweep),
  E3 (acceptance drift). Writes CSVs + PNGs to `results/`.
- `vllm_integration.py` — how the controller connects to real vLLM V1
  spec-decode: Tier 1 (batch→γ schedule, implemented) and Tier 2 (online
  proposer patch, stretch).
- `hw/` — the hardware study run on PSC (V100, vLLM 0.11.0, fp16, Triton
  backend, Qwen2.5-7B, n-gram spec decode):
  - `setup_psc.sh`, `check_env.py`, `step1_smoke.py` — env install + probe + smoke test
  - `step2_baseline.py` — fixed-γ acceptance/throughput logging
  - `step3_calibrate.py` — fits `cost_model` constants from real decode microbenchmarks
  - `step4_grid.py` — measured (γ × batch) throughput/acceptance grid
  - `step5_policies.py` — no-spec / static / adaptive / adaptive+off / oracle comparison + plots
  - `step6_findings.py` — generates `results_hw/FINDINGS.md` from the measured CSVs

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
cd project && ../.venv/bin/python run_experiments.py
```

## What the results show (simulation)
- The throughput-optimal speculation length falls from γ≈4 at batch 1 to γ=1
  once the batch is large — no single static γ is best across load.
- The adaptive controller tracks the per-step oracle and beats every static
  setting across the load sweep on both throughput and goodput.
- Under within-generation acceptance drift, the controller lowers γ as
  acceptance decays, beating static-γ baselines.

## What the hardware study shows (measured on a PSC V100)
See `results_hw/FINDINGS.md` (auto-generated from the measured CSVs).
- Cost model **calibrated** on real decode microbenchmarks: `t_mem≈22.9 ms`,
  `t_flop≈0.145 ms/tok` (R²≈0.94) → V100 decode is **memory-bound**, the regime
  the project assumes.
- Speculation is **load-dependent**: n-gram gives ~+20% throughput at batch 1
  but −5% at batch 64; the throughput-optimal γ **falls from 4 to 1** as batch
  grows — no single static γ is best across load.
- The **adaptive** controller has the **lowest worst-case throughput shortfall**
  of any fixed policy across the load sweep.

## Remaining / stretch
- EAGLE (trained drafter) for higher acceptance → wider adaptive advantage.
- Tier-2 online per-step γ (vLLM proposer patch); cost model that captures
  per-spec-step overhead (so it turns speculation off at high batch).
- Server-based Poisson/ShareGPT arrival sweep for req/s goodput under a TPOT SLO.
