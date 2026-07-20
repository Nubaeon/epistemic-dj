"""Bayesian conjugate-Gaussian belief updating -- standalone reimplementation
of Empirica's own core/bayesian_beliefs.py::BayesianBeliefManager math (same
formulas, no empirica package dependency, per the standalone-product
decision). Precision-weighted average of prior and observation; posterior
variance shrinks with accumulated evidence. Chosen over isotonic regression
for the two corrections built on top of this (CalibrationStore's term-bias
and margin-scale beliefs) because it degrades gracefully at small n -- the
prior dominates until real evidence accumulates, rather than overfitting
noise the way a monotonic step-function fit would at n~6-18.
"""

from __future__ import annotations

from epistemic_dj.models import Belief

DEFAULT_OBSERVATION_VARIANCE = 0.1  # matches Empirica's own default


def update_belief(
    prior_mean: float,
    prior_variance: float,
    evidence_count: int,
    observation: float,
    obs_variance: float = DEFAULT_OBSERVATION_VARIANCE,
) -> Belief:
    posterior_mean = (prior_variance * observation + obs_variance * prior_mean) / (
        prior_variance + obs_variance
    )
    posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / obs_variance)
    return Belief(
        mean=posterior_mean, variance=posterior_variance, evidence_count=evidence_count + 1
    )
