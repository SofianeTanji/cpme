import numpy as np
from core import LongitudinalOPEData
from policies import GaussianPolicy, MixturePolicy


def _feedback(L_t, A_t, noise_std=0.5):
    """L_{t+1} = 0.5 * L_t + 0.3 * A_t + noise  (treatment-confounder feedback)."""
    return 0.5 * L_t + 0.3 * A_t[:, np.newaxis] + noise_std * np.random.randn(*L_t.shape)


def make_longitudinal_scenario(scenario_id, K=1, d=5, seed=None):
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
    if seed is not None:
        np.random.seed(seed)

    w_base = np.ones(d) / np.sqrt(d)
    beta = np.linspace(0.1, 0.5, d)
    treatment_effect = 1.0

    logging_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]

    if scenario_id == "I":
        pi_policies       = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        pi_prime_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]

    elif scenario_id == "II":
        shift = 2.0 * np.ones(d)
        pi_policies       = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        pi_prime_policies = [GaussianPolicy(w_base + shift, scale=1.0) for _ in range(K + 1)]

    elif scenario_id == "III":
        w1 = w_base + np.ones(d)
        w2 = w_base - np.ones(d)
        pi_policies       = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        pi_prime_policies = [
            MixturePolicy(GaussianPolicy(w1, scale=1.0), GaussianPolicy(w2, scale=1.0))
            for _ in range(K + 1)
        ]

    elif scenario_id == "IV":
        # π_t = π′_t = GaussianPolicy(w_base) at all stages, but π′ at t≥1
        # will be passed L_0 (lagged) instead of L_t by the data generator.
        # We mark π′ policies with a special attribute so the generator knows.
        pi_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        pi_prime_policies = [GaussianPolicy(w_base, scale=1.0) for _ in range(K + 1)]
        # Tag stages t>=1 of pi_prime to use lagged covariate
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
    seed=None,
) -> LongitudinalOPEData:
    """
    Simulate longitudinal logged data and compute all quantities needed for NestedDRKPT.

    Parameters
    ----------
    n                   : number of trajectories
    logging_policies    : list of K+1 GaussianPolicy (one per stage)
    pi_policies         : list of K+1 policies for π
    pi_prime_policies   : list of K+1 policies for π′
    beta, treatment_effect : outcome model coefficients
    K                   : number of stages (default 1)
    noise_std           : std of outcome noise

    Returns
    -------
    LongitudinalOPEData
    """
    if seed is not None:
        np.random.seed(seed)

    d = logging_policies[0].w.shape[0]

    L = [None] * (K + 1)
    A = [None] * (K + 1)
    g_props = [None] * (K + 1)      # logging propensities at each stage

    W_pi       = np.ones((K + 1, n))
    W_pi_prime = np.ones((K + 1, n))
    pi_samples       = [None] * (K + 1)
    pi_prime_samples = [None] * (K + 1)

    # Initial covariates
    L[0] = np.random.randn(n, d)
    L0 = L[0]  # keep reference for Scenario IV lagged conditioning

    for t in range(K + 1):
        # Logging action and propensity
        A[t] = logging_policies[t].sample_treatments(L[t])
        g_props[t] = logging_policies[t].get_propensities(L[t], A[t])

        # Cumulative IS weights (product up to stage t)
        pi_t_props = pi_policies[t].get_propensities(L[t], A[t])

        # For π′: Scenario IV uses lagged covariate L_0 at t >= 1
        L_for_prime = L0 if getattr(pi_prime_policies[t], "_use_lagged", False) else L[t]
        pi_prime_t_props = pi_prime_policies[t].get_propensities(L_for_prime, A[t])

        if t == 0:
            W_pi[t]       = pi_t_props / g_props[t]
            W_pi_prime[t] = pi_prime_t_props / g_props[t]
        else:
            W_pi[t]       = W_pi[t - 1] * pi_t_props / g_props[t]
            W_pi_prime[t] = W_pi_prime[t - 1] * pi_prime_t_props / g_props[t]

        # Policy integration samples (one draw per observation)
        pi_samples[t]       = pi_policies[t].sample_treatments(L[t])
        pi_prime_L          = L0 if getattr(pi_prime_policies[t], "_use_lagged", False) else L[t]
        pi_prime_samples[t] = pi_prime_policies[t].sample_treatments(pi_prime_L)

        # Feedback: generate next-stage covariates
        if t < K:
            L[t + 1] = _feedback(L[t], A[t])

    # Terminal outcome
    Y = L[K] @ beta + treatment_effect * A[K] + noise_std * np.random.randn(n)

    return LongitudinalOPEData(
        L=L,
        A=A,
        Y=Y,
        W_pi=W_pi,
        W_pi_prime=W_pi_prime,
        pi_samples=pi_samples,
        pi_prime_samples=pi_prime_samples,
    )
