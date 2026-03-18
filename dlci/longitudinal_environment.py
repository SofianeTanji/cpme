import numpy as np
from dlci.core import LongitudinalOPEData
from dlci.policies import GaussianPolicy, MixturePolicy


def _feedback(L_t, A_t, rng: np.random.Generator, noise_std=0.5):
    """L_{t+1} = 0.5 * L_t + 0.3 * A_t + noise  (treatment-confounder feedback)."""
    return (
        0.5 * L_t
        + 0.3 * A_t[:, np.newaxis]
        + noise_std * rng.standard_normal(L_t.shape)
    )


def make_longitudinal_scenario(scenario_id, K=1, d=5):
    """
    Returns (logging_policies, pi_policies, pi_prime_policies, beta, treatment_effect).
    Each policy list has length K+1 (one Markov policy per stage, acting on L_t).

    Scenarios:
      I   — Null: π = π′ at every stage, feedback present.
      II  — Mean shift: π′_t uses weight vector shifted by +2 at each stage.
      III — Mixture: π′_t is a bimodal MixturePolicy at each stage.
      IV  — Feedback-only shift: π_t = π′_t at t=0; at t=1 π′_1 conditions on L_0
            instead of L_1, so marginals of A_1 match but the feedback-response differs.
            Non-nested methods are blind to this; the nested test is consistent.
    """
    w_base = np.ones(d) / np.sqrt(d)
    beta = np.linspace(0.1, 0.5, d)
    treatment_effect = 1.0

    logging_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]

    if scenario_id == "I":
        pi_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        pi_prime_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]

    elif scenario_id == "II":
        shift = 2.0 * np.ones(d)
        pi_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        pi_prime_policies = [
            GaussianPolicy(w_base + shift, scale=1.0) for _ in range(K + 1)
        ]

    elif scenario_id == "III":
        w1 = w_base + np.ones(d)
        w2 = w_base - np.ones(d)
        pi_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        pi_prime_policies = [
            MixturePolicy(GaussianPolicy(w1, scale=1.0), GaussianPolicy(w2, scale=1.0))
            for _ in range(K + 1)
        ]

    elif scenario_id == "IV":
        # π_t = π′_t in form, but π′ at t≥1 receives L_0 instead of L_t.
        # The _use_lagged flag signals the data generator to pass the lagged covariate.
        pi_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        pi_prime_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        for t in range(1, K + 1):
            pi_prime_policies[t]._use_lagged = True

    else:
        raise ValueError(f"Unknown scenario '{scenario_id}'")

    return logging_policies, pi_policies, pi_prime_policies, beta, treatment_effect


def generate_longitudinal_ope_data(
    n,
    logging_policies,
    pi_policies,
    pi_prime_policies,
    beta,
    treatment_effect,
    K=1,
    noise_std=0.1,
    rng=None,
) -> LongitudinalOPEData:
    rng = np.random.default_rng(rng)
    d = logging_policies[0].w.shape[0]

    L = [None] * (K + 1)
    A = [None] * (K + 1)
    g_props = [None] * (K + 1)

    W_pi = np.ones((K + 1, n))
    W_pi_prime = np.ones((K + 1, n))
    pi_samples = [None] * (K + 1)
    pi_prime_samples = [None] * (K + 1)

    L[0] = rng.standard_normal((n, d))
    L0 = L[0]  # kept for Scenario IV: π′ at t≥1 conditions on L_0 instead of L_t

    for t in range(K + 1):
        A[t] = logging_policies[t].sample_treatments(L[t], rng)
        g_props[t] = logging_policies[t].get_propensities(L[t], A[t])

        pi_t_props = pi_policies[t].get_propensities(L[t], A[t])

        # Scenario IV: π′ at t≥1 uses lagged covariate L_0
        L_for_prime = (
            L0 if getattr(pi_prime_policies[t], "_use_lagged", False) else L[t]
        )
        pi_prime_t_props = pi_prime_policies[t].get_propensities(L_for_prime, A[t])

        if t == 0:
            W_pi[t] = pi_t_props / g_props[t]
            W_pi_prime[t] = pi_prime_t_props / g_props[t]
        else:
            W_pi[t] = W_pi[t - 1] * pi_t_props / g_props[t]
            W_pi_prime[t] = W_pi_prime[t - 1] * pi_prime_t_props / g_props[t]

        pi_samples[t] = pi_policies[t].sample_treatments(L[t], rng)
        pi_prime_L = L0 if getattr(pi_prime_policies[t], "_use_lagged", False) else L[t]
        pi_prime_samples[t] = pi_prime_policies[t].sample_treatments(pi_prime_L, rng)

        if t < K:
            L[t + 1] = _feedback(L[t], A[t], rng)

    Y = L[K] @ beta + treatment_effect * A[K] + noise_std * rng.standard_normal(n)

    return LongitudinalOPEData(
        L=L,
        A=A,
        Y=Y,
        W_pi=W_pi,
        W_pi_prime=W_pi_prime,
        pi_samples=pi_samples,
        pi_prime_samples=pi_prime_samples,
    )
