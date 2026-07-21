import pytest

from epistemic_dj.calibration.bayesian_belief import beta_belief, beta_update, update_belief


def test_posterior_mean_between_prior_and_observation():
    belief = update_belief(prior_mean=0.0, prior_variance=0.1, evidence_count=0, observation=1.0)
    assert 0.0 < belief.mean < 1.0


def test_posterior_variance_shrinks_from_prior():
    belief = update_belief(prior_mean=0.0, prior_variance=0.1, evidence_count=0, observation=1.0)
    assert belief.variance < 0.1


def test_evidence_count_increments():
    belief = update_belief(prior_mean=0.0, prior_variance=0.1, evidence_count=5, observation=0.5)
    assert belief.evidence_count == 6


def test_repeated_consistent_observations_converge_toward_observation():
    mean, variance, count = 0.0, 0.25, 0
    for _ in range(20):
        belief = update_belief(mean, variance, count, observation=0.8)
        mean, variance, count = belief.mean, belief.variance, belief.evidence_count
    assert mean == pytest.approx(0.8, abs=0.05)


def test_single_observation_does_not_fully_overwrite_prior():
    # With a tight prior and one loose observation, the posterior shouldn't
    # jump all the way to the observation -- that would defeat the point of
    # having a prior at all (degrades to "just use the last data point").
    belief = update_belief(
        prior_mean=0.0, prior_variance=0.01, evidence_count=10, observation=1.0, obs_variance=1.0
    )
    assert belief.mean < 0.5


def test_beta_update_uninformative_prior_starts_at_half():
    belief = beta_belief(1.0, 1.0)
    assert belief.mean == pytest.approx(0.5)
    assert belief.evidence_count == 0


def test_beta_update_success_increments_alpha():
    alpha, beta = beta_update(1.0, 1.0, 1)
    assert (alpha, beta) == (2.0, 1.0)
    belief = beta_belief(alpha, beta)
    assert belief.mean == pytest.approx(2 / 3)
    assert belief.evidence_count == 1


def test_beta_update_failure_increments_beta():
    alpha, beta = beta_update(1.0, 1.0, 0)
    assert (alpha, beta) == (1.0, 2.0)
    belief = beta_belief(alpha, beta)
    assert belief.mean == pytest.approx(1 / 3)


def test_beta_belief_converges_toward_true_hit_rate():
    alpha, beta = 1.0, 1.0
    # 8 successes, 2 failures -- true rate 0.8. With the Beta(1,1) prior's
    # implicit pseudo-observations folded in, posterior mean is 9/12=0.75,
    # not exactly 0.8 -- that's the prior doing its job, not an error.
    for observation in [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]:
        alpha, beta = beta_update(alpha, beta, observation)
    belief = beta_belief(alpha, beta)
    assert belief.mean == pytest.approx(0.75)
    assert belief.evidence_count == 10


def test_beta_belief_variance_shrinks_with_evidence():
    prior_variance = beta_belief(1.0, 1.0).variance
    alpha, beta = 1.0, 1.0
    for _ in range(10):
        alpha, beta = beta_update(alpha, beta, 1)
    assert beta_belief(alpha, beta).variance < prior_variance
