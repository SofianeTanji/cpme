import numpy as np
from core import PolicyTest, OPEData
from kernels import build_kernel_matrix


class KPT(PolicyTest):
    def __init__(self, kernel_function="rbf", gamma=None, iterations=1000, random_state=None):
        self.kernel_function = kernel_function
        self.gamma = gamma
        self.iterations = iterations
        self.random_state = random_state

    def test(self, data: OPEData) -> dict:
        """
        Returns {"stat": float, "null": np.ndarray, "pval": float}.
        """
        Y = data.Y.reshape(-1, 1) if data.Y.ndim == 1 else data.Y
        w_pi = data.w_pi.ravel()
        w_pi_prime = data.w_pi_prime.ravel()

        K = self._build_kernel(Y)
        stat = self._compute_mmd2(K, w_pi, w_pi_prime)
        null = self._compute_null(K, w_pi, w_pi_prime)
        return {"stat": float(stat), "null": null, "pval": float(np.mean(null > stat))}

    def _build_kernel(self, Y: np.ndarray) -> np.ndarray:
        """Applies median heuristic automatically when kernel='rbf' and gamma=None."""
        return build_kernel_matrix(Y, metric=self.kernel_function, gamma=self.gamma)

    def _compute_mmd2(self, K: np.ndarray, w_pi: np.ndarray, w_pi_prime: np.ndarray) -> float:
        """Single authoritative MMD² formula."""
        n = len(w_pi)
        K_pi = np.outer(w_pi, w_pi) * K
        K_pi_prime = np.outer(w_pi_prime, w_pi_prime) * K
        K_cross = np.outer(w_pi, w_pi_prime) * K
        return (
            (K_pi.sum() - np.trace(K_pi)) / (n * (n - 1))
            + (K_pi_prime.sum() - np.trace(K_pi_prime)) / (n * (n - 1))
            - 2 * K_cross.sum() / (n ** 2)
        )

    def _compute_null(self, K: np.ndarray, w_pi: np.ndarray, w_pi_prime: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        n = len(w_pi)
        null = np.zeros(self.iterations)
        for i in range(self.iterations):
            idx = rng.permutation(n)
            null[i] = self._compute_mmd2(K, w_pi[idx], w_pi_prime[idx])
        return null
