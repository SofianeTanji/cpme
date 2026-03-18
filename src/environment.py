import numpy as np
from core import OPEData
from policies import GaussianPolicy, MixturePolicy, EstimatedLoggingPolicy


def outcome_model(X, T, beta, treatment_effect, noise_std=0.1):
    noise = noise_std * np.random.randn(len(T))
    return X @ beta + treatment_effect * T + noise


def generate_ope_data(
    X, policy_logging, policy_pi, policy_pi_prime, beta, treatment_effect, noise_std=0.1
) -> OPEData:
    """
    Simulate logged bandit data and prepare inputs for counterfactual evaluation.

    Parameters:
        X                  : Covariates (n, d)
        policy_logging     : Policy that generated the logged data (π₀)
        policy_pi          : Target policy π
        policy_pi_prime    : Alternative policy π′
        beta               : Outcome main-effect parameters
        treatment_effect   : Treatment effect coefficient
        noise_std          : Std of additive Gaussian noise

    Returns:
        OPEData dataclass with fields {X, T, Y, w_pi, w_pi_prime, pi_samples, pi_prime_samples}
    """
    T = policy_logging.sample_treatments(X)
    Y = outcome_model(X, T, beta, treatment_effect, noise_std)
    estimate_logging_propensities = EstimatedLoggingPolicy(X, T).get_propensities(X, T)
    w_pi = (
        policy_pi.get_propensities(X, T)[:, np.newaxis]
        / estimate_logging_propensities[:, np.newaxis]
    )
    w_pi_prime = (
        policy_pi_prime.get_propensities(X, T)[:, np.newaxis]
        / estimate_logging_propensities[:, np.newaxis]
    )

    pi_samples = policy_pi.sample_treatments(X)
    pi_prime_samples = policy_pi_prime.sample_treatments(X)

    return OPEData(
        X=X,
        T=T,
        Y=Y,
        w_pi=w_pi,
        w_pi_prime=w_pi_prime,
        pi_samples=pi_samples,
        pi_prime_samples=pi_prime_samples,
    )


def make_scenario(scenario_id, d=5, seed=None):
    """
    Returns:
        - X: covariates
        - policy_logging: logging policy π₀
        - policy_pi: target policy π
        - policy_pi_prime: alternative policy π′
        - beta, treatment_effect: for outcome model
    """
    if seed is not None:
        np.random.seed(seed)

    n = 1000  # number of samples
    X = np.random.normal(0, 1, size=(n, d))

    beta = np.linspace(0.1, 0.5, d)  # outcome main effect
    treatment_effect = 1.0            # treatment effect coefficient

    w_base = np.ones(d) / np.sqrt(d)

    # Logging policy: fixed
    policy_logging = GaussianPolicy(w_base, scale=1)

    if scenario_id == "I":
        # Null scenario: π = π′ (no difference)
        policy_pi = GaussianPolicy(w_base, scale=1)
        policy_pi_prime = GaussianPolicy(w_base, scale=1)

    elif scenario_id == "II":
        # Mean shift: π′ mean shifted along one direction
        shift = 2 * np.ones(d)
        policy_pi = GaussianPolicy(w_base, scale=1.0)
        policy_pi_prime = GaussianPolicy(w_base + shift, scale=1.0)

    elif scenario_id == "III":
        # Mixture policy π′: bimodal
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
