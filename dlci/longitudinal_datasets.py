import numpy as np
from dlci.core import OPEData, LongitudinalOPEData
from dlci.policies import EstimatedLoggingPolicy
from dlci.longitudinal_environment import (
    make_longitudinal_scenario,
    generate_longitudinal_ope_data,
)


def longitudinal_to_ope_data(data: LongitudinalOPEData) -> OPEData:
    # Adapter for applying non-longitudinal methods (KPT, DRKPT) as baselines on longitudinal
    # data. These methods were not designed for the longitudinal problem: they apply a single-stage
    # DR correction at the terminal stage only, while earlier stages are handled through the
    # cumulative IS weights W_pi[K] = prod_{t=0}^{K} pi_t(A_t|L_t) / g_t(A_t|L_t).
    return OPEData(
        L=data.L[-1],
        A=data.A[-1],
        Y=data.Y,
        w_pi=data.W_pi[[-1]].T,  # (n, 1) cumulative IS weight at terminal stage
        w_pi_prime=data.W_pi_prime[[-1]].T,
        pi_samples=data.pi_samples[-1],
        pi_prime_samples=data.pi_prime_samples[-1],
    )


class RealLongitudinalDataset:
    """Pre-collected longitudinal logged data with multiple stages.

    Parameters
    ----------
    L_list : list of ndarray
        Covariates at each stage [L_0, ..., L_K], each shape (n, d).
    A_list : list of ndarray
        Treatments at each stage [A_0, ..., A_K], each shape (n,).
    Y : ndarray, shape (n,)
        Terminal outcome.
    logging_propensities : list of ndarray or None
        Known logging propensities [g_0, ..., g_K], each shape (n,).
        If None, estimated via EstimatedLoggingPolicy at each stage.
    seed : int or None
    """

    def __init__(
        self,
        L_list: list,
        A_list: list,
        Y: np.ndarray,
        logging_propensities: list | None = None,
        seed=None,
    ):
        self.L = L_list
        self.A = A_list
        self.Y = Y
        self.K = len(L_list) - 1
        self._logging_propensities = logging_propensities
        self._rng = np.random.default_rng(seed)

    def prepare_ope_data(self, pi_policies, pi_prime_policies) -> LongitudinalOPEData:
        K = self.K
        n = len(self.Y)

        W_pi = np.ones((K + 1, n))
        W_pi_prime = np.ones((K + 1, n))
        pi_samples = [None] * (K + 1)
        pi_prime_samples = [None] * (K + 1)

        for t in range(K + 1):
            L_t, A_t = self.L[t], self.A[t]

            if self._logging_propensities is not None:
                g_t = self._logging_propensities[t]
            else:
                g_t = EstimatedLoggingPolicy(L_t, A_t).get_propensities(L_t, A_t)

            pi_t = pi_policies[t].get_propensities(L_t, A_t)
            pi_prime_t = pi_prime_policies[t].get_propensities(L_t, A_t)

            if t == 0:
                W_pi[t] = pi_t / g_t
                W_pi_prime[t] = pi_prime_t / g_t
            else:
                W_pi[t] = W_pi[t - 1] * pi_t / g_t
                W_pi_prime[t] = W_pi_prime[t - 1] * pi_prime_t / g_t

            pi_samples[t] = pi_policies[t].sample_treatments(L_t, self._rng)
            pi_prime_samples[t] = pi_prime_policies[t].sample_treatments(L_t, self._rng)

        return LongitudinalOPEData(
            L=self.L,
            A=self.A,
            Y=self.Y,
            W_pi=W_pi,
            W_pi_prime=W_pi_prime,
            pi_samples=pi_samples,
            pi_prime_samples=pi_prime_samples,
        )


class LongitudinalSyntheticDataset:
    """Wraps make_longitudinal_scenario + generate_longitudinal_ope_data."""

    def __init__(
        self, scenario_id: str, K: int = 1, ns: int = 200, d: int = 5, seed=None
    ):
        self.scenario_id = scenario_id
        self.K = K
        self.ns = ns
        self.d = d
        self._rng = np.random.default_rng(seed)

        (
            self.logging_policies,
            self.pi_policies,
            self.pi_prime_policies,
            self._beta,
            self._te,
        ) = make_longitudinal_scenario(scenario_id, K=K, d=d)

    def prepare_ope_data(self, pi_policies, pi_prime_policies) -> LongitudinalOPEData:
        return generate_longitudinal_ope_data(
            n=self.ns,
            logging_policies=self.logging_policies,
            pi_policies=pi_policies,
            pi_prime_policies=pi_prime_policies,
            beta=self._beta,
            treatment_effect=self._te,
            K=self.K,
            rng=self._rng,
        )
