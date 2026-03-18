import numpy as np
from scipy.stats import norm
from core import LongitudinalOPEData


def _log_gaussian_propensity(A, L, coeff):
    """log N(A; coeff * L, 1) for scalar A and L (both shape (n,))."""
    mu = coeff * L.ravel()
    return norm.logpdf(A.ravel(), loc=mu, scale=1.0)


def generate_dsprites_data(
    n,
    Delta=0.0,
    logging_coeff=0.3,
    eval_coeff=0.5,
    noise_std_feedback=1.0,
    noise_std_outcome=0.1,
    seed=None,
) -> LongitudinalOPEData:
    """
    Generate LongitudinalOPEData for the dSprites-inspired DR experiment (K=1).

    DGP
    ---
    L1 ~ N(0, 1)                                  shape (n, 1)
    A1 ~ N(logging_coeff * L1, 1)                 shape (n,)
    L2 = 0.5*L1 + 0.5*A1 + N(0, noise_std_feedback)  shape (n, 1)
    A2 ~ N(logging_coeff * L2, 1)                 shape (n,)
    Y  = [L2, sin(L2), A2, cos(A2), L2*A2] + eps  shape (n, 5)

    Behavioral logging policy g_t: N(logging_coeff * L_t, 1)
    Evaluation policies:
        pi_e,t  : N(eval_coeff * L_t, 1)
        pi_e',t : N((eval_coeff + Delta) * L_t, 1)

    IS weights are computed in log-space for numerical stability.
    At Delta=0 (null): pi_e = pi_e', so H0 is strictly true.
    Because eval_coeff != logging_coeff, weights are non-trivial even at Delta=0.
    """
    if seed is not None:
        np.random.seed(seed)

    # --- DGP ---
    L1 = np.random.randn(n, 1)
    A1 = logging_coeff * L1.ravel() + np.random.randn(n)

    L2 = 0.5 * L1 + 0.5 * A1[:, np.newaxis] + noise_std_feedback * np.random.randn(n, 1)
    A2 = logging_coeff * L2.ravel() + np.random.randn(n)

    l2 = L2.ravel()
    Y = np.column_stack([l2, np.sin(l2), A2, np.cos(A2), l2 * A2])
    Y += noise_std_outcome * np.random.randn(n, 5)

    # --- IS weights in log-space ---
    log_g1  = _log_gaussian_propensity(A1, L1, logging_coeff)
    log_pi1 = _log_gaussian_propensity(A1, L1, eval_coeff)
    log_pp1 = _log_gaussian_propensity(A1, L1, eval_coeff + Delta)

    log_g2  = _log_gaussian_propensity(A2, L2, logging_coeff)
    log_pi2 = _log_gaussian_propensity(A2, L2, eval_coeff)
    log_pp2 = _log_gaussian_propensity(A2, L2, eval_coeff + Delta)

    W_pi       = np.zeros((2, n))
    W_pi_prime = np.zeros((2, n))
    W_pi[0]       = np.exp(log_pi1 - log_g1)
    W_pi_prime[0] = np.exp(log_pp1 - log_g1)
    W_pi[1]       = W_pi[0]       * np.exp(log_pi2 - log_g2)
    W_pi_prime[1] = W_pi_prime[0] * np.exp(log_pp2 - log_g2)

    # --- Policy integration samples (one draw per observation per stage) ---
    pi_samples0  = eval_coeff         * L1.ravel() + np.random.randn(n)
    pip_samples0 = (eval_coeff + Delta) * L1.ravel() + np.random.randn(n)
    pi_samples1  = eval_coeff         * L2.ravel() + np.random.randn(n)
    pip_samples1 = (eval_coeff + Delta) * L2.ravel() + np.random.randn(n)

    return LongitudinalOPEData(
        L=[L1, L2],
        A=[A1, A2],
        Y=Y,
        W_pi=W_pi,
        W_pi_prime=W_pi_prime,
        pi_samples=[pi_samples0, pi_samples1],
        pi_prime_samples=[pip_samples0, pip_samples1],
    )


def recompute_weights(
    data: LongitudinalOPEData,
    logging_coeff_wrong=0.0,
    eval_coeff=0.5,
    Delta=0.0,
) -> LongitudinalOPEData:
    """
    Return a new LongitudinalOPEData with IS weights recomputed using a misspecified
    logging policy (logging_coeff_wrong, e.g. 0.0 for intercept-only).

    All other fields (L, A, Y, pi_samples, pi_prime_samples) are unchanged.
    Used to implement propensity misspecification in the DR grid.
    """
    L1, L2 = data.L
    A1, A2 = data.A
    n = len(data.Y)

    log_g1_w  = _log_gaussian_propensity(A1, L1, logging_coeff_wrong)
    log_pi1   = _log_gaussian_propensity(A1, L1, eval_coeff)
    log_pp1   = _log_gaussian_propensity(A1, L1, eval_coeff + Delta)

    log_g2_w  = _log_gaussian_propensity(A2, L2, logging_coeff_wrong)
    log_pi2   = _log_gaussian_propensity(A2, L2, eval_coeff)
    log_pp2   = _log_gaussian_propensity(A2, L2, eval_coeff + Delta)

    W_pi       = np.zeros((2, n))
    W_pi_prime = np.zeros((2, n))
    W_pi[0]       = np.exp(log_pi1 - log_g1_w)
    W_pi_prime[0] = np.exp(log_pp1 - log_g1_w)
    W_pi[1]       = W_pi[0]       * np.exp(log_pi2 - log_g2_w)
    W_pi_prime[1] = W_pi_prime[0] * np.exp(log_pp2 - log_g2_w)

    return LongitudinalOPEData(
        L=data.L,
        A=data.A,
        Y=data.Y,
        W_pi=W_pi,
        W_pi_prime=W_pi_prime,
        pi_samples=data.pi_samples,
        pi_prime_samples=data.pi_prime_samples,
    )
