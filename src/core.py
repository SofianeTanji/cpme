from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class Policy(ABC):
    @abstractmethod
    def sample_treatments(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_propensities(self, X: np.ndarray, t: np.ndarray) -> np.ndarray: ...


class PolicyTest(ABC):
    @abstractmethod
    def test(self, data: "OPEData") -> dict: ...


@dataclass
class OPEData:
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    w_pi: np.ndarray
    w_pi_prime: np.ndarray
    pi_samples: np.ndarray
    pi_prime_samples: np.ndarray


@dataclass
class LongitudinalOPEData:
    """Longitudinal trajectory data for K+1 stages (t = 0, ..., K).

    L[t]               : (n, d) covariates at stage t  (Markov state)
    A[t]               : (n,)   actions at stage t
    Y                  : (n,)   terminal outcome
    W_pi[t, i]         : cumulative IS weight ∏_{s=0}^t π_s(A_s|L_s) / g_{s,0}(A_s|L_s)
    W_pi_prime[t, i]   : same for π′
    pi_samples[t]      : (n,) one action draw ã_i ~ π_t(·|L_{t,i}), for policy integration
    pi_prime_samples[t]: same for π′
    """
    L               : list          # list of (n, d) arrays, length K+1
    A               : list          # list of (n,) arrays,   length K+1
    Y               : np.ndarray    # (n,)
    W_pi            : np.ndarray    # (K+1, n)
    W_pi_prime      : np.ndarray    # (K+1, n)
    pi_samples      : list          # list of (n,) arrays,   length K+1
    pi_prime_samples: list          # list of (n,) arrays,   length K+1
