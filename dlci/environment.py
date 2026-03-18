import numpy as np
from dlci.core import OPEData
from dlci.policies import GaussianPolicy, MixturePolicy, EstimatedLoggingPolicy


def outcome_model(
    L, A, beta, treatment_effect, rng: np.random.Generator, noise_std=0.1
):
    noise = noise_std * rng.standard_normal(len(A))
    return L @ beta + treatment_effect * A + noise


def generate_ope_data(
    L,
    policy_logging,
    policy_pi,
    policy_pi_prime,
    beta,
    treatment_effect,
    rng=None,
    noise_std=0.1,
) -> OPEData:
    rng = np.random.default_rng(rng)
    A = policy_logging.sample_treatments(L, rng)
    Y = outcome_model(L, A, beta, treatment_effect, rng, noise_std)
    g_hat = EstimatedLoggingPolicy(L, A).get_propensities(L, A)
    w_pi = policy_pi.get_propensities(L, A)[:, np.newaxis] / g_hat[:, np.newaxis]
    w_pi_prime = (
        policy_pi_prime.get_propensities(L, A)[:, np.newaxis] / g_hat[:, np.newaxis]
    )

    pi_samples = policy_pi.sample_treatments(L, rng)
    pi_prime_samples = policy_pi_prime.sample_treatments(L, rng)

    return OPEData(
        L=L,
        A=A,
        Y=Y,
        w_pi=w_pi,
        w_pi_prime=w_pi_prime,
        pi_samples=pi_samples,
        pi_prime_samples=pi_prime_samples,
    )


def make_scenario(scenario_id, d=5, rng=None):
    rng = np.random.default_rng(rng)
    n = 1000
    X = rng.standard_normal((n, d))

    beta = np.linspace(0.1, 0.5, d)
    treatment_effect = 1.0

    w_base = np.ones(d) / np.sqrt(d)
    policy_logging = GaussianPolicy(w_base, scale=1)

    if scenario_id == "I":
        policy_pi = GaussianPolicy(w_base, scale=1)
        policy_pi_prime = GaussianPolicy(w_base, scale=1)

    elif scenario_id == "II":
        # π and π′ shifted symmetrically in opposite directions
        shift = np.ones(d)
        policy_pi = GaussianPolicy(w_base - shift, scale=1.0)
        policy_pi_prime = GaussianPolicy(w_base + shift, scale=1.0)

    elif scenario_id == "III":
        # π′ is a bimodal mixture; π is unimodal Gaussian
        w1 = w_base + np.ones(d)
        w2 = w_base - np.ones(d)
        policy_pi = GaussianPolicy(w_base)
        policy_pi_prime = MixturePolicy(GaussianPolicy(w1), GaussianPolicy(w2))

    elif scenario_id == "IV":
        w1 = w_base + 2 * np.ones(d)
        w2 = w_base
        policy_pi = GaussianPolicy(w_base)
        policy_pi_prime = MixturePolicy(GaussianPolicy(w1), GaussianPolicy(w2))

    else:
        raise ValueError(f"Unknown scenario {scenario_id}")

    return X, policy_logging, policy_pi, policy_pi_prime, beta, treatment_effect
