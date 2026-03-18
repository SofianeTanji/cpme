import numpy as np
from core import LongitudinalOPEData
from longitudinal_environment import make_longitudinal_scenario, generate_longitudinal_ope_data


class LongitudinalSyntheticDataset:
    """Wraps make_longitudinal_scenario + generate_longitudinal_ope_data.

    Mirrors the interface of SyntheticDataset in datasets.py.

    Parameters
    ----------
    scenario_id : str
        "I", "II", "III", or "IV"
    K           : int
        Number of stages (default 1)
    ns          : int
        Number of trajectories
    d           : int
        Covariate dimension
    seed        : int or None
    """

    def __init__(self, scenario_id: str, K: int = 1, ns: int = 200, d: int = 5, seed=None):
        self.scenario_id = scenario_id
        self.K = K
        self.ns = ns
        self.d = d
        self.seed = seed

        (
            self.logging_policies,
            self.pi_policies,
            self.pi_prime_policies,
            self._beta,
            self._te,
        ) = make_longitudinal_scenario(scenario_id, K=K, d=d, seed=seed)

    def prepare_ope_data(self) -> LongitudinalOPEData:
        return generate_longitudinal_ope_data(
            n=self.ns,
            logging_policies=self.logging_policies,
            pi_policies=self.pi_policies,
            pi_prime_policies=self.pi_prime_policies,
            beta=self._beta,
            treatment_effect=self._te,
            K=self.K,
            seed=self.seed,
        )
