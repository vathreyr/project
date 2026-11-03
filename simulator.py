"""
simulator.py
------------
A discrete-event simulator of an LLM server doing speculative decoding under
continuous batching. It is intentionally a *coarse* model: the point is to
study how the best speculation length changes with load and to compare
policies, not to predict absolute A100 numbers. All timing comes from
cost_model.HardwareModel; all policy decisions come from controller.py.

What it models:
  * Poisson request arrivals at rate lambda (req/s).
  * Per-request output lengths (so requests finish and leave the batch).
  * Continuous batching with a max batch cap; a queue when overloaded.
  * Heterogeneous per-request acceptance rates, with optional within-request
    drift (acceptance decays as a generation gets longer).
  * Realistic accept sampling: gamma Bernoulli(alpha) draws, accepted = length
    of the leading run of successes, tokens emitted = accepted + 1 bonus.

What it reports per run:
  throughput (tok/s), mean/p50/p90/p99 TPOT (s/token), SLO attainment,
  goodput (req/s meeting the TPOT SLO), mean batch size, mean gamma.
"""

from __future__ import annotations
import heapq
import numpy as np
from dataclasses import dataclass, field

from cost_model import HardwareModel
from controller import OraclePolicy


@dataclass
class Request:
    rid: int
    arrival: float
    output_len: int
    base_alpha: float            # acceptance rate at the start of generation
    drift: float = 0.0           # per-token linear decay of acceptance
    produced: int = 0
    decode_time: float = 0.0     # cumulative time this request spent decoding
    admit_time: float = -1.0
    finish_time: float = -1.0

    def current_alpha(self) -> float:
        a = self.base_alpha - self.drift * self.produced
        return float(min(0.98, max(0.05, a)))

    @property
    def remaining(self) -> int:
        return self.output_len - self.produced


@dataclass
class RunResult:
    label: str
    lam: float
    throughput: float
    tpot_mean: float
    tpot_p50: float
    tpot_p90: float
    tpot_p99: float
    slo_attainment: float
    goodput: float
    mean_batch: float
    mean_gamma: float
    n_requests: int
    sim_time: float


@dataclass
class WorkloadConfig:
    n_requests: int = 2000
    lam: float = 20.0                 # arrival rate (req/s)
    mean_output: int = 220            # mean decode length (tokens)
    alpha_mean: float = 0.72          # mean per-request acceptance
    alpha_spread: float = 0.12        # heterogeneity across requests
    drift: float = 0.0                # within-request acceptance decay / token
    seed: int = 0


def sample_workload(cfg: WorkloadConfig) -> list[Request]:
    rng = np.random.default_rng(cfg.seed)
    # Poisson arrivals: exponential inter-arrival gaps.
    gaps = rng.exponential(1.0 / cfg.lam, size=cfg.n_requests)
    arrivals = np.cumsum(gaps)
    # Output lengths: lognormal, clipped, so there is a realistic tail.
    out = rng.lognormal(mean=np.log(cfg.mean_output), sigma=0.5,
                        size=cfg.n_requests)
    out = np.clip(out, 16, 1024).astype(int)
    # Per-request acceptance: Beta scaled to (alpha_mean +/- spread).
    a = rng.normal(cfg.alpha_mean, cfg.alpha_spread, size=cfg.n_requests)
    a = np.clip(a, 0.35, 0.95)
    return [Request(rid=i, arrival=float(arrivals[i]), output_len=int(out[i]),
                    base_alpha=float(a[i]), drift=cfg.drift)
            for i in range(cfg.n_requests)]


def simulate(policy, cfg: WorkloadConfig, hw: HardwareModel,
             max_batch: int = 64, slo_tpot: float = 0.04) -> RunResult:
    """
    Run one policy against one workload. slo_tpot is the per-output-token
    latency target (s/token); a request "meets SLO" if its average TPOT is
    below it.
    """
    rng = np.random.default_rng(cfg.seed + 12345)
    reqs = sample_workload(cfg)
    reqs.sort(key=lambda r: r.arrival)

    arrival_idx = 0
    running: list[Request] = []
    completed: list[Request] = []
    clock = 0.0
    total_tokens = 0
    batch_accum = 0.0
    gamma_accum = 0.0
    n_steps = 0

    n = len(reqs)
    while arrival_idx < n or running:
        # Admit everything that has arrived by `clock`, up to the batch cap.
        while (arrival_idx < n and reqs[arrival_idx].arrival <= clock
               and len(running) < max_batch):
            r = reqs[arrival_idx]
            r.admit_time = clock
            running.append(r)
            arrival_idx += 1

        # If nothing is running, jump the clock to the next arrival.
        if not running:
            if arrival_idx < n:
                clock = reqs[arrival_idx].arrival
                continue
            else:
                break

        batch = len(running)
        # Oracle needs the true mean acceptance of the current batch.
        if isinstance(policy, OraclePolicy):
            policy.set_true_alpha(float(np.mean([r.current_alpha()
                                                 for r in running])))
        gamma = policy.choose_gamma(batch, n_steps)
        t_step = hw.macro_step_seconds(batch, gamma)

        accepted_this_step = []
        still_running = []
        for r in running:
            r.decode_time += t_step
            if gamma > 0:
                draws = rng.random(gamma) < r.current_alpha()
                # leading run of accepted draft tokens
                acc = int(np.argmin(draws)) if not draws.all() else gamma
                emitted = acc + 1            # + bonus token
            else:
                acc = 0
                emitted = 1
            emitted = min(emitted, r.remaining)
            r.produced += emitted
            total_tokens += emitted
            accepted_this_step.append(acc)
            if r.remaining <= 0:
                r.finish_time = clock + t_step
                completed.append(r)
            else:
                still_running.append(r)

        # Feed the realized acceptance back to the policy as the fraction of
        # draft tokens accepted this step across the batch (in [0, 1]).
        if gamma > 0 and accepted_this_step:
            frac = float(np.sum(accepted_this_step)) / (gamma * batch)
        else:
            frac = 0.0
        policy.observe(frac, gamma)

        running = still_running
        clock += t_step
        batch_accum += batch
        gamma_accum += gamma
        n_steps += 1

    sim_time = clock
    tpots = np.array([r.decode_time / max(1, r.produced) for r in completed])
    meets = tpots <= slo_tpot
    return RunResult(
        label=policy.name,
        lam=cfg.lam,
        throughput=total_tokens / sim_time if sim_time > 0 else 0.0,
        tpot_mean=float(tpots.mean()),
        tpot_p50=float(np.percentile(tpots, 50)),
        tpot_p90=float(np.percentile(tpots, 90)),
        tpot_p99=float(np.percentile(tpots, 99)),
        slo_attainment=float(meets.mean()),
        goodput=float(meets.sum() / sim_time) if sim_time > 0 else 0.0,
        mean_batch=batch_accum / max(1, n_steps),
        mean_gamma=gamma_accum / max(1, n_steps),
        n_requests=len(completed),
        sim_time=sim_time,
    )
