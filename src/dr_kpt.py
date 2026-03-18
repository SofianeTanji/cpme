import numpy as np
from core import PolicyTest, OPEData
from kernels import median_bandwidth, build_kernel_matrix, build_cross_kernel_matrix, tune_reg_lambda as _tune_reg_lambda, cross_ustat


class DRKPT(PolicyTest):
    def __init__(self, kernel_function="rbf", reg_lambda=1e-2, cross_fit=False):
        self.kernel_function = kernel_function
        self.reg_lambda = reg_lambda
        self.cross_fit = cross_fit

    def test(self, data: OPEData) -> dict:
        """Returns {"stat": float, "pval": float}. pval = 1 - norm.cdf(stat)."""
        if self.cross_fit:
            return self._test_cross_fit(data)
        return self._test_full(data)

    @staticmethod
    def tune_reg_lambda(X, T, Y, reg_grid=None, num_cv=3) -> float:
        return _tune_reg_lambda(X, T, Y, reg_grid=reg_grid, num_cv=num_cv)

    def _setup_kernels(self, X, T, pi_samples, pi_prime_samples, bw_rows=None) -> dict:
        """
        Computes KX, KT, KT_pi, KT_pi_prime.
        bw_rows: index slice for bandwidth estimation (None=all; slice(N//2, None) for CF).
        Returns dict of kernel matrices + scalar bandwidths (bw_x, bw_t).
        """
        X_bw = X[bw_rows] if bw_rows is not None else X
        T_bw = T[bw_rows] if bw_rows is not None else T

        bw_x = median_bandwidth(X_bw)
        bw_t = median_bandwidth(T_bw[:, np.newaxis])

        gamma_x = 1.0 / bw_x
        gamma_t = 1.0 / bw_t

        T_col = T[:, np.newaxis]
        pi_col = pi_samples[:, np.newaxis]
        pi_prime_col = pi_prime_samples[:, np.newaxis]

        KX = build_kernel_matrix(X, metric="rbf", gamma=gamma_x)
        KT = build_kernel_matrix(T_col, metric="rbf", gamma=gamma_t)
        KT_pi = build_cross_kernel_matrix(T_col, pi_col, metric="rbf", gamma=gamma_t)
        KT_pi_prime = build_cross_kernel_matrix(T_col, pi_prime_col, metric="rbf", gamma=gamma_t)

        return dict(KX=KX, KT=KT, KT_pi=KT_pi, KT_pi_prime=KT_pi_prime,
                    bw_x=bw_x, bw_t=bw_t)

    def _compute_dr_term(self, KX, KT, KT_pi, KT_pi_prime, w_pi, w_pi_prime) -> np.ndarray:
        """Solves 3 KRR systems; forms DR correction. Uses self.reg_lambda."""
        N = KX.shape[0]
        reg = self.reg_lambda
        A = np.multiply(KX, KT) + reg * np.eye(N)
        mu_logging = np.linalg.solve(A, np.multiply(KX, KT))
        mu_pi = np.linalg.solve(A, np.multiply(KX, KT_pi))
        mu_pi_prime = np.linalg.solve(A, np.multiply(KX, KT_pi_prime))
        return mu_pi_prime - mu_pi + (w_pi_prime - w_pi) * (np.eye(N) - mu_logging)

    def _test_full(self, data: OPEData) -> dict:
        Y = data.Y.reshape(-1, 1) if data.Y.ndim == 1 else data.Y
        w_pi = data.w_pi
        w_pi_prime = data.w_pi_prime

        kernels = self._setup_kernels(data.X, data.T, data.pi_samples, data.pi_prime_samples)
        dr_term = self._compute_dr_term(
            kernels["KX"], kernels["KT"], kernels["KT_pi"], kernels["KT_pi_prime"],
            w_pi, w_pi_prime
        )
        KY = build_kernel_matrix(Y, metric=self.kernel_function)
        prod = dr_term.T @ KY @ dr_term
        return cross_ustat(prod)

    def _test_cross_fit(self, data: OPEData) -> dict:
        Y = data.Y.reshape(-1, 1) if data.Y.ndim == 1 else data.Y
        w_pi = data.w_pi
        w_pi_prime = data.w_pi_prime
        N = len(Y)
        N2 = N // 2

        # Bandwidth estimated on second half (matching original xMMD2dr_cross_fit)
        kernels = self._setup_kernels(
            data.X, data.T, data.pi_samples, data.pi_prime_samples,
            bw_rows=slice(N2, None)
        )
        KX = kernels["KX"]
        KT = kernels["KT"]
        KT_pi = kernels["KT_pi"]
        KT_pi_prime = kernels["KT_pi_prime"]

        # Split 1: solve on first half
        reg = self.reg_lambda
        A1 = np.multiply(KX[:N2, :N2], KT[:N2, :N2]) + reg * np.eye(N2)
        mu_logging_1 = np.linalg.solve(A1, np.multiply(KX[:N2, :N2], KT[:N2, :N2]))
        mu_pi_1 = np.linalg.solve(A1, np.multiply(KX[:N2, :N2], KT_pi[:N2, :N2]))
        mu_pi_prime_1 = np.linalg.solve(A1, np.multiply(KX[:N2, :N2], KT_pi_prime[:N2, :N2]))

        # Split 2: solve on second half
        A2 = np.multiply(KX[N2:, N2:], KT[N2:, N2:]) + reg * np.eye(N2)
        mu_logging_2 = np.linalg.solve(A2, np.multiply(KX[N2:, N2:], KT[N2:, N2:]))
        mu_pi_2 = np.linalg.solve(A2, np.multiply(KX[N2:, N2:], KT_pi[N2:, N2:]))
        mu_pi_prime_2 = np.linalg.solve(A2, np.multiply(KX[N2:, N2:], KT_pi_prime[N2:, N2:]))

        left = (
            mu_pi_prime_1 - mu_pi_1
            + (w_pi_prime[:N2] - w_pi[:N2]) * (np.eye(N2) - mu_logging_1)
        )
        right = (
            mu_pi_prime_2 - mu_pi_2
            + (w_pi_prime[N2:] - w_pi[N2:]) * (np.eye(N2) - mu_logging_2)
        )

        KY = build_cross_kernel_matrix(Y[:N2], Y[N2:], metric=self.kernel_function)
        prod = left.T @ KY @ right
        return cross_ustat(prod)
