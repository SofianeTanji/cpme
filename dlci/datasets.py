from abc import ABC, abstractmethod
import numpy as np
from dlci.core import OPEData, Policy
from dlci.policies import EstimatedLoggingPolicy


class Dataset(ABC):
    @abstractmethod
    def prepare_ope_data(self, policy_pi: Policy, policy_pi_prime: Policy) -> OPEData:
        """Compute importance weights and counterfactual samples, return OPEData."""
        ...


class SyntheticDataset(Dataset):
    """Wraps make_scenario + generate_ope_data for a fixed scenario/seed/size."""

    def __init__(self, scenario_id: str, ns: int, d: int = 5, seed: int | None = None):
        from dlci.environment import make_scenario

        self._rng = np.random.default_rng(seed)
        (
            X,
            self.policy_logging,
            self.policy_pi,
            self.policy_pi_prime,
            self._beta,
            self._te,
        ) = make_scenario(scenario_id, d=d, rng=self._rng)
        self.L = X[:ns]

    def prepare_ope_data(self, policy_pi: Policy, policy_pi_prime: Policy) -> OPEData:
        from dlci.environment import generate_ope_data

        return generate_ope_data(
            self.L,
            self.policy_logging,
            policy_pi,
            policy_pi_prime,
            self._beta,
            self._te,
            rng=self._rng,
        )


class RealDataset(Dataset):
    """Pre-collected logged data (X, T, Y) with optional known logging propensities."""

    def __init__(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        logging_propensities: np.ndarray | None = None,
        seed=None,
    ):
        self.L = X
        self.A = T
        self.Y = Y
        self._logging_propensities = logging_propensities
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_csv(
        cls,
        path: str,
        x_cols,
        t_col: str,
        y_col: str,
        propensity_col: str | None = None,
        seed=None,
    ) -> "RealDataset":
        """Load from a CSV file. x_cols: list of covariate column names."""
        import pandas as pd

        df = pd.read_csv(path)
        X = df[x_cols].to_numpy(dtype=float)
        T = df[t_col].to_numpy(dtype=float)
        Y = df[y_col].to_numpy(dtype=float)
        prop = df[propensity_col].to_numpy(dtype=float) if propensity_col else None
        return cls(X, T, Y, prop, seed=seed)

    def prepare_ope_data(self, policy_pi: Policy, policy_pi_prime: Policy) -> OPEData:
        L, A, Y = self.L, self.A, self.Y
        if self._logging_propensities is not None:
            log_props = self._logging_propensities
        else:
            log_props = EstimatedLoggingPolicy(L, A).get_propensities(L, A)
        w_pi = (policy_pi.get_propensities(L, A) / log_props)[:, np.newaxis]
        w_pi_prime = (policy_pi_prime.get_propensities(L, A) / log_props)[:, np.newaxis]
        return OPEData(
            L=L,
            A=A,
            Y=Y,
            w_pi=w_pi,
            w_pi_prime=w_pi_prime,
            pi_samples=policy_pi.sample_treatments(L, self._rng),
            pi_prime_samples=policy_pi_prime.sample_treatments(L, self._rng),
        )
