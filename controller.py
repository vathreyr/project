"""
controller.py
-------------
Speculation-length policies. These are the objects under study. Each policy
exposes a single method:

    choose_gamma(batch_size, step_index) -> int

and is told the outcome of each step via observe(accepted_counts, gamma) so
that adaptive policies can update their acceptance estimate online.

The AdaptiveController is written so the same logic can be lifted into vLLM's
spec-decode path: it only needs (a) the current running batch size, (b) a
cheap online estimate of the acceptance rate, and (c) the cost model. None of
that requires anything vLLM does not already have at scheduling time.
"""

from __future__ import annotations
from cost_model import HardwareModel, expected_tokens_per_step


class StaticPolicy:
    """Fixed speculation length -- the status quo in today's serving stacks."""

    def __init__(self, gamma: int):
        self.gamma = gamma
        self.name = f"static-g{gamma}"

    def choose_gamma(self, batch_size: int, step_index: int) -> int:
        return self.gamma

    def observe(self, accepted_fraction: float, gamma: int) -> None:
        pass


class NoSpecPolicy(StaticPolicy):
    """gamma = 0: ordinary autoregressive decoding (one token per step)."""

    def __init__(self):
        super().__init__(0)
        self.name = "no-spec"


class AdaptiveController:
    """
    Goodput-aware adaptive speculation length.

    Keeps an exponential-moving-average (EWMA) estimate of the per-token
    acceptance rate from realized accept/reject outcomes, and at each step
    picks the gamma that maximizes a proxy for goodput: expected accepted
    tokens per unit of GPU time at the *current* batch size,

        gamma* = argmax_g  E[tokens | alpha_hat, g] / t_step(batch, g)

    Because t_step grows with batch*(g+1) (see cost_model), the denominator
    penalizes long speculation exactly when the batch is large and the GPU is
    compute-bound -- which is the behaviour we want.
    """

    def __init__(self, hw: HardwareModel, gamma_max: int = 8,
                 ewma_decay: float = 0.9, alpha_init: float = 0.7):
        self.hw = hw
        self.gamma_max = gamma_max
        self.ewma_decay = ewma_decay
        self.alpha_hat = alpha_init
        self.name = "adaptive"

    def _best_gamma(self, batch_size: int, alpha: float) -> int:
        best_g, best_score = 1, -1.0
        for g in range(1, self.gamma_max + 1):
            toks = expected_tokens_per_step(alpha, g)
            cost = self.hw.macro_step_seconds(batch_size, g)
            score = toks / cost  # accepted tokens per second (per request)
            if score > best_score:
                best_score, best_g = score, g
        return best_g

    def choose_gamma(self, batch_size: int, step_index: int) -> int:
        return self._best_gamma(batch_size, self.alpha_hat)

    def observe(self, alpha_obs: float, gamma: int) -> None:
        """
        Update the per-token acceptance estimate from one step's outcome.

        `alpha_obs` is an observed-trial estimate of the per-token acceptance
        probability, that is accepted_tokens / observed_trials, where an
        observed trial is a draft position that was actually resolved (every
        accepted token, plus the single rejection that terminated the run).
        It addresses the normalization bias of accepted_tokens / (gamma*batch):
        under leading-run acceptance that ratio is biased low (about 0.47 at
        true alpha=0.72, gamma=4) because positions after the first rejection
        are never tested, which would push the controller toward shorter
        speculation. The observed-trial ratio is not a provably unbiased
        estimator in finite samples (for gamma=2 on one sequence its
        expectation is alpha + alpha(1-alpha)/2), and pooling trials across a
        heterogeneous batch weights requests by how many trials they produced.
        The caller (simulator / vLLM hook) forms alpha_obs.
        """
        if gamma <= 0:
            return
        realized = min(max(alpha_obs, 0.0), 1.0)
        self.alpha_hat = (self.ewma_decay * self.alpha_hat
                          + (1.0 - self.ewma_decay) * realized)


class BatchAwareFixedAlpha:
    """
    Ablation baseline that isolates the contribution of ONLINE acceptance
    estimation. It is batch-aware (it picks gamma from the current batch size
    through the same cost-model objective) but uses a FIXED acceptance rate and
    never updates it from feedback. Comparing the full AdaptiveController
    against this policy on a drifting-acceptance workload isolates exactly how
    much the online alpha estimate contributes, which the static-gamma
    baselines cannot.
    """

    def __init__(self, hw: HardwareModel, gamma_max: int = 8,
                 alpha_fixed: float = 0.7):
        self.hw = hw
        self.gamma_max = gamma_max
        self.alpha_fixed = alpha_fixed
        self.name = f"batch-only(a={alpha_fixed})"

    def choose_gamma(self, batch_size: int, step_index: int) -> int:
        best_g, best = 1, -1.0
        for g in range(1, self.gamma_max + 1):
            score = (expected_tokens_per_step(self.alpha_fixed, g)
                     / self.hw.macro_step_seconds(batch_size, g))
            if score > best:
                best, best_g = score, g
        return best_g

    def observe(self, alpha_obs: float, gamma: int) -> None:
        pass


class OraclePolicy:
    """
    Reference policy: at every step, picks the gamma that maximizes the SAME
    tokens-per-second proxy but using the request's TRUE current acceptance
    rate rather than an online estimate. It is not realizable in practice (it
    needs the true current acceptance) and is not a proven upper bound on
    long-run SLO goodput; it is a reference point for how much the online
    acceptance estimate costs the adaptive controller.
    """

    def __init__(self, hw: HardwareModel, gamma_max: int = 8):
        self.hw = hw
        self.gamma_max = gamma_max
        self.name = "oracle"
        self._true_alpha = 0.7  # set by the simulator each step

    def set_true_alpha(self, alpha: float) -> None:
        self._true_alpha = alpha

    def choose_gamma(self, batch_size: int, step_index: int) -> int:
        best_g, best_score = 1, -1.0
        for g in range(1, self.gamma_max + 1):
            toks = expected_tokens_per_step(self._true_alpha, g)
            cost = self.hw.macro_step_seconds(batch_size, g)
            score = toks / cost
            if score > best_score:
                best_score, best_g = score, g
        return best_g

    def observe(self, accepted_fraction: float, gamma: int) -> None:
        pass
