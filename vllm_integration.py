"""
vllm_integration.py
--------------------
How the AdaptiveController connects to a REAL vLLM speculative-decode run.

This replaces the earlier 0.6.x-V0 sketch (`num_lookahead_slots` /
`SpecDecodeWorker`), which no longer exists: vLLM removed the V0 engine in
0.11.0, and the V1 spec-decode path is organized completely differently. The
integration we actually use on PSC (V100, vLLM 0.11.0, n-gram spec decode) has
two tiers, both reusing `controller.py` unchanged.

-----------------------------------------------------------------------------
TIER 1 -- controller-driven batch->gamma schedule (implemented, measured)
-----------------------------------------------------------------------------
vLLM 0.11.0 fixes `num_speculative_tokens` at engine-construction time, so the
realizable "adaptive" policy is a schedule that maps the running batch size to
a speculation length. The schedule is produced by the SAME objective the
simulator's AdaptiveController maximizes -- expected accepted tokens per unit
of GPU time under the calibrated cost model -- and the acceptance it is fed is
the real per-step acceptance FRACTION measured from vLLM's spec-decode metrics
(`vllm:spec_decode_num_accepted_tokens / num_draft_tokens`), exactly the float
contract in `controller.AdaptiveController.observe`.

This is what `hw/step4_grid.py` (measure the gamma x batch grid, reading the
acceptance fraction) and `hw/step5_policies.py` (choose gamma per batch with
the controller objective) implement end to end. The function below is the
shared decision rule; step5 applies it per batch.
-----------------------------------------------------------------------------
TIER 2 -- true per-step online gamma (stretch; GPU-only, not yet landed)
-----------------------------------------------------------------------------
To change gamma live inside one running engine, patch the V1 n-gram proposer
(`vllm.v1.spec_decode.ngram_proposer.NgramProposer`) so the number of proposed
tokens is `AdaptiveController.choose_gamma(running_batch, step)` each step, and
feed the rejection sampler's accepted counts back via `observe(fraction, g)`.
The mixin below marks those two hook points. This needs a vLLM source patch and
can only be validated on the GPU, so it is kept separate from the Tier-1 path
that produced the reported numbers.
"""

from __future__ import annotations

from cost_model import HardwareModel, expected_tokens_per_step
from controller import AdaptiveController


# ---------------------------------------------------------------------------
# TIER 1: the decision rule (reused by hw/step5_policies.py).
# ---------------------------------------------------------------------------
def controller_gamma_for_batch(hw: HardwareModel, batch: int, alpha_hat: float,
                               candidate_gammas) -> int:
    """
    The AdaptiveController objective, restricted to a set of candidate gammas we
    can actually run (e.g. the ones benchmarked, optionally including 0 = no
    speculation). Identical maths to AdaptiveController._best_gamma; the
    candidate set is passed in so the same rule covers both the faithful
    (gamma >= 1) and the "may turn speculation off" (gamma >= 0) variants.
    """
    best_g, best = candidate_gammas[0], -1.0
    for g in candidate_gammas:
        score = expected_tokens_per_step(alpha_hat, g) / hw.macro_step_seconds(batch, g)
        if score > best:
            best, best_g = score, g
    return best_g


# ---------------------------------------------------------------------------
# TIER 2: online hooks (stretch goal -- requires a vLLM V1 proposer patch).
# ---------------------------------------------------------------------------
class AdaptiveSpecDecodeMixin:
    """
    Intended to sit on vLLM's V1 n-gram/EAGLE proposer. The two hooks are the
    only contact points the controller needs; the controller object itself is
    the unmodified simulator one, so sim and hardware stay consistent.
    """

    def __init__(self, *args, gamma_max: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctrl = AdaptiveController(HardwareModel.load_calibrated(),
                                        gamma_max=gamma_max)

    # hook 1: how many tokens to speculate for the current running batch.
    def adaptive_num_spec_tokens(self, running_batch: int) -> int:
        return self._ctrl.choose_gamma(running_batch, step_index=0)

    # hook 2: feed the rejection sampler's accepted counts back as a FRACTION
    # (float -- rounding to an int biases the estimate toward the minimum gamma;
    # see controller.AdaptiveController.observe).
    def on_verify_complete(self, num_accepted: int, num_drafted: int,
                           gamma: int) -> None:
        if gamma <= 0 or num_drafted <= 0:
            return
        self._ctrl.observe(num_accepted / num_drafted, gamma)
