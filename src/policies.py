import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from core import Policy


class GaussianPolicy(Policy):
    def __init__(self, w, scale=1.0):
        self.w = w
        self.scale = scale

    def sample_treatments(self, X: np.ndarray) -> np.ndarray:
        return np.random.normal(self.get_mean(X), self.scale)

    def get_mean(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def get_propensities(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        mean = self.get_mean(X)
        return (1 / (self.scale * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((t - mean) / self.scale) ** 2
        )


class MixturePolicy(Policy):
    def __init__(self, policy1: Policy, policy2: Policy):
        self.p1 = policy1
        self.p2 = policy2

    def sample_treatments(self, X: np.ndarray) -> np.ndarray:
        mask = np.random.binomial(1, 0.5, size=X.shape[0])
        T1 = self.p1.sample_treatments(X)
        T2 = self.p2.sample_treatments(X)
        return mask * T1 + (1 - mask) * T2

    def get_propensities(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        return 0.5 * self.p1.get_propensities(X, t) + 0.5 * self.p2.get_propensities(X, t)


class EstimatedLoggingPolicy(Policy):
    def __init__(self, X: np.ndarray, T: np.ndarray):
        if np.issubdtype(T.dtype, np.integer) and np.unique(T).size == 2:
            self.model_type = "binary"
            self.model = LogisticRegression()
            self.model.fit(X, ((T + 1) // 2))  # map {-1, +1} to {0, 1}
        else:
            self.model_type = "continuous"
            self.model = LinearRegression()
            self.model.fit(X, T)

    def sample_treatments(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_propensities(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.model_type == "binary":
            prob = self.model.predict_proba(X)[:, 1]
            t = ((t + 1) // 2).astype(int)
            return np.where(t == 1, prob, 1 - prob)
        else:
            mean = self.model.predict(X)
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (t - mean) ** 2)

    def get_mean(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
