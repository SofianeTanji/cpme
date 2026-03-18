import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from dlci.core import Policy


class GaussianPolicy(Policy):
    def __init__(self, w, scale=1.0):
        self.w = w
        self.scale = scale
        self._use_lagged: bool = False

    def sample_treatments(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.get_mean(X), self.scale)

    def get_mean(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def get_propensities(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return norm.pdf(t, loc=self.get_mean(X), scale=self.scale)


class MixturePolicy(Policy):
    def __init__(self, policy1: Policy, policy2: Policy):
        self.p1 = policy1
        self.p2 = policy2

    def sample_treatments(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mask = rng.integers(0, 2, size=X.shape[0])
        T1 = self.p1.sample_treatments(X, rng)
        T2 = self.p2.sample_treatments(X, rng)
        return mask * T1 + (1 - mask) * T2

    def get_propensities(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return 0.5 * self.p1.get_propensities(X, t) + 0.5 * self.p2.get_propensities(
            X, t
        )


class EstimatedLoggingPolicy(Policy):
    def __init__(self, X: np.ndarray, T: np.ndarray):
        self.model = LinearRegression()
        self.model.fit(X, T)

    def sample_treatments(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.model.predict(X)

    def get_propensities(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        mean = self.model.predict(X)
        return norm.pdf(t, loc=mean, scale=1.0)

    def get_mean(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
