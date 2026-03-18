import numpy as np
from scipy.linalg import lu_factor, lu_solve
from dlci.core import PolicyTest, OPEData
from dlci.kernels import (
    median_bandwidth,
    build_kernel_matrix,
    build_cross_kernel_matrix,
    tune_reg_lambda as _tune_reg_lambda,
    cross_ustat,
)


class DRKPT(PolicyTest):
    def __init__(self, kernel_function="rbf", reg_lambda=1e-2, cross_fit=False):
        self.kernel_function = kernel_function
        self.reg_lambda = reg_lambda
        self.cross_fit = cross_fit

    def test(self, data: OPEData, KY=None) -> dict:
        """Returns {"stat": float, "pval": float}. pval = 1 - norm.cdf(stat).

        KY : optional precomputed (n, n) outcome kernel matrix. If None, computed
             internally from data.Y using self.kernel_function.
        """
        if self.cross_fit:
            return self._test_cross_fit(data, KY=KY)
        return self._test_full(data, KY=KY)

    def compute_outcome_kernel(self, Y: np.ndarray) -> np.ndarray:
        """Build the full (n, n) outcome kernel matrix from Y.

        Call this once for expensive kernels, then pass the result to test().
        """
        Y_k = Y if Y.ndim > 1 else Y.reshape(-1, 1)
        return build_kernel_matrix(Y_k, metric=self.kernel_function)

    @staticmethod
    def tune_reg_lambda(L, A, Y, reg_grid=None, num_cv=3) -> float:
        return _tune_reg_lambda(L, A, Y, reg_grid=reg_grid, num_cv=num_cv)

    def _setup_kernels(self, L, A, pi_samples, pi_prime_samples, bw_rows=None):
        """Returns (KL, KA, KA_pi, KA_pi_prime). bw_rows restricts bandwidth estimation."""
        L_bw = L[bw_rows] if bw_rows is not None else L
        A_bw = A[bw_rows] if bw_rows is not None else A

        gamma_l = 1.0 / median_bandwidth(L_bw)
        gamma_a = 1.0 / median_bandwidth(A_bw[:, np.newaxis])

        A_col = A[:, np.newaxis]
        pi_col = pi_samples[:, np.newaxis]
        pi_prime_col = pi_prime_samples[:, np.newaxis]

        KL = build_kernel_matrix(L, metric="rbf", gamma=gamma_l)
        KA = build_kernel_matrix(A_col, metric="rbf", gamma=gamma_a)
        KA_pi = build_cross_kernel_matrix(A_col, pi_col, metric="rbf", gamma=gamma_a)
        KA_pi_prime = build_cross_kernel_matrix(
            A_col, pi_prime_col, metric="rbf", gamma=gamma_a
        )

        return KL, KA, KA_pi, KA_pi_prime

    def _compute_dr_term(
        self, KL, KA, KA_pi, KA_pi_prime, w_pi, w_pi_prime
    ) -> np.ndarray:
        N = KL.shape[0]
        M = np.multiply(KL, KA) + self.reg_lambda * np.eye(N)
        M_lu = lu_factor(M)
        mu_logging = lu_solve(M_lu, np.multiply(KL, KA))
        mu_pi = lu_solve(M_lu, np.multiply(KL, KA_pi))
        mu_pi_prime = lu_solve(M_lu, np.multiply(KL, KA_pi_prime))
        return mu_pi_prime - mu_pi + (w_pi_prime - w_pi) * (np.eye(N) - mu_logging)

    def _test_full(self, data: OPEData, KY=None) -> dict:
        Y = data.Y.reshape(-1, 1) if data.Y.ndim == 1 else data.Y
        w_pi = data.w_pi
        w_pi_prime = data.w_pi_prime

        KL, KA, KA_pi, KA_pi_prime = self._setup_kernels(
            data.L, data.A, data.pi_samples, data.pi_prime_samples
        )
        dr_term = self._compute_dr_term(KL, KA, KA_pi, KA_pi_prime, w_pi, w_pi_prime)
        if KY is None:
            KY = build_kernel_matrix(Y, metric=self.kernel_function)
        prod = dr_term.T @ KY @ dr_term
        return cross_ustat(prod)

    def _test_cross_fit(self, data: OPEData, KY=None) -> dict:
        Y = data.Y.reshape(-1, 1) if data.Y.ndim == 1 else data.Y
        w_pi = data.w_pi
        w_pi_prime = data.w_pi_prime
        N = len(Y)
        N2 = N // 2

        # Bandwidth estimated on second half (matching original xMMD2dr_cross_fit)
        KL, KA, KA_pi, KA_pi_prime = self._setup_kernels(
            data.L,
            data.A,
            data.pi_samples,
            data.pi_prime_samples,
            bw_rows=slice(N2, None),
        )

        left = self._compute_dr_term(
            KL[:N2, :N2],
            KA[:N2, :N2],
            KA_pi[:N2, :N2],
            KA_pi_prime[:N2, :N2],
            w_pi[:N2],
            w_pi_prime[:N2],
        )
        right = self._compute_dr_term(
            KL[N2:, N2:],
            KA[N2:, N2:],
            KA_pi[N2:, N2:],
            KA_pi_prime[N2:, N2:],
            w_pi[N2:],
            w_pi_prime[N2:],
        )

        if KY is not None:
            KY_cross = KY[:N2, N2:]
        else:
            KY_cross = build_cross_kernel_matrix(
                Y[:N2], Y[N2:], metric=self.kernel_function
            )  # bandwidth estimated from Y[:N2] only
        prod = left.T @ KY_cross @ right
        return cross_ustat(prod)
