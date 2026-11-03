"""
cost_model.py
-------------
A simple roofline-style latency model for a single speculative-decoding
macro-step (one propose-then-verify round) under continuous batching.

The model deliberately has very few parameters so that its behaviour is easy
to reason about and so that it can later be *calibrated* against a handful of
real vLLM measurements on PSC (a couple of microbenchmarks are enough to fit
t_mem and t_flop). The qualitative behaviour we care about -- a target
forward pass being nearly free at small token counts and becoming linear in
token count once the GPU is compute-bound -- is exactly what a memory-bound /
compute-bound roofline predicts, and is what motivates the whole project.

NOTE: the default constants below are order-of-magnitude values chosen to sit
in the regime reported for ~7-8B decoding on an A100-class GPU. They are
illustrative, not measured. Calibrating them is part of the remaining
hardware work (see the report).
"""

import json
import os
from dataclasses import dataclass


@dataclass
class HardwareModel:
    # --- target model (large) single forward pass: t = t_mem + t_flop * n ---
    # t_mem: memory-bound floor of one decode step in seconds (reading the
    #        weights + KV dominates here, independent of how many tokens we
    #        push through). ~12 ms is a plausible 8B/A100 decode floor.
    t_mem_target: float = 12.0e-3
    # t_flop: marginal seconds per *token* processed once compute-bound.
    t_flop_target: float = 0.11e-3

    # --- draft model (small) one token for a batch of B requests ---
    # The draft is cheap per token but must be run gamma times sequentially.
    t_mem_draft: float = 1.3e-3
    t_flop_draft: float = 0.012e-3

    def target_pass_seconds(self, n_tokens: int) -> float:
        """Latency of ONE target verification pass over n_tokens positions."""
        return self.t_mem_target + self.t_flop_target * n_tokens

    def draft_step_seconds(self, batch: int) -> float:
        """Latency of drafting ONE token for `batch` requests."""
        return self.t_mem_draft + self.t_flop_draft * batch

    def macro_step_seconds(self, batch: int, gamma: int) -> float:
        """
        Wall-clock time of a full propose(gamma)+verify macro-step for a
        batch of `batch` requests.

        - Drafting: gamma sequential small-model steps over `batch` requests.
        - Verifying: one target pass over batch*(gamma+1) positions
          (gamma speculative + 1 bonus position per request).
        """
        draft = gamma * self.draft_step_seconds(batch)
        verify = self.target_pass_seconds(batch * (gamma + 1))
        return draft + verify

    @classmethod
    def load_calibrated(cls, path: str | None = None) -> "HardwareModel":
        """
        Return a HardwareModel using constants measured on real hardware, if a
        calibration file exists; otherwise fall back to the illustrative
        defaults above. The calibration JSON is produced by
        hw/step3_calibrate.py and read by both the vLLM path and the simulator,
        so sim and hardware optimize over the SAME cost model.

        The JSON only needs to carry the four constants; any missing key keeps
        its default.
        """
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results_hw", "calibrated_constants.json")
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        c = data.get("constants", data)  # accept either {constants:{...}} or flat
        hw = cls()
        for k in ("t_mem_target", "t_flop_target",
                  "t_mem_draft", "t_flop_draft"):
            if k in c and c[k] is not None:
                setattr(hw, k, float(c[k]))
        return hw


def expected_tokens_per_step(alpha: float, gamma: int) -> float:
    """
    Expected number of accepted tokens emitted per request per macro-step,
    following the standard speculative-sampling analysis (Leviathan et al.,
    2023): with per-token acceptance probability alpha and gamma draft tokens,
    E[tokens] = (1 - alpha^(gamma+1)) / (1 - alpha).
    This already includes the guaranteed bonus token from the target.
    """
    if gamma <= 0:
        return 1.0
    if alpha >= 1.0:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)
