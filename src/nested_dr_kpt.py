import numpy as np
from core import PolicyTest, LongitudinalOPEData
from kernels import median_bandwidth, build_kernel_matrix, build_cross_kernel_matrix, tune_reg_lambda as _tune_reg_lambda, cross_ustat


class NestedDRKPT(PolicyTest):
    """Nested doubly-robust kernel policy test for longitudinal data (K >= 1 stages).

    The test statistic is a cross U-statistic asymptotically N(0,1) under H_0.

    The core computation maintains a coefficient matrix V (m x m) where V[k, i] is the
    coefficient of phi_Y(Y_k) in the H_Y-valued function Q^pi_t evaluated at H_{t,i}.
    This tracks how the terminal outcome kernel φ_Y(Y) propagates backward through stages.

    Starting from V_{K+1} = I (terminal: Q^pi_{K+1} = phi_Y(Y)), at each stage t:
        - Residual coefficient : V_res = V @ (I - Alpha_t)
        - Policy integration   : V_next = V @ (C_t @ Gamma_t)^T
    where Alpha_t = Gamma_t @ B_t, Gamma_t = (B_t + m*lambda*I)^{-1}, B_t = KH_t * KA_t,
    C_t = KH_t * KAp_t (cross kernel using pi samples vs observed actions).

    The DR coefficient vector for observation i is then:
        D[k, i] = sum_t W_t(i) * V_res_t[k, i]  +  V_0[k, i]   (Q^pi_0 base term)

    The EIF difference Phi_{pi,pi'} has coefficient D = D_pi - D_pi_prime, and the
    inner product matrix is prod = D.T @ KY_cross @ D (identical structure to DRKPT).
    """

    def __init__(self, reg_lambda: float = 1e-2, cross_fit: bool = True, state_kernel: str = "rbf"):
        self.reg_lambda = reg_lambda
        self.cross_fit = cross_fit
        self.state_kernel = state_kernel

    def test(self, data: LongitudinalOPEData) -> dict:
        """Returns {"stat": float, "pval": float}."""
        if self.cross_fit:
            return self._test_cross_fit(data)
        return self._test_full(data)

    # ------------------------------------------------------------------
    # Public utility
    # ------------------------------------------------------------------

    @staticmethod
    def tune_reg_lambda(L_last, A_last, Y, reg_grid=None, num_cv=3) -> float:
        return _tune_reg_lambda(L_last, A_last, Y, reg_grid=reg_grid, num_cv=num_cv)

    # ------------------------------------------------------------------
    # DR coefficient matrix
    # ------------------------------------------------------------------

    def _compute_dr_coeff(self, L, A, Y, pi_samples, pi_prime_samples, W_pi, W_pi_prime):
        """
        Compute the (m, m) DR coefficient matrix D = D_pi - D_pi_prime where:

            D_pi[k, i] = coefficient of phi_Y(Y_k) in Phi_pi(O_i)

        Backward induction:
            V_{K+1} = I  (terminal: Q^pi_{K+1}(H_{K+1,i}) = phi_Y(Y_i))

            for t = K, ..., 0:
                B_t     = KH_t * KA_t
                Alpha_t = (B_t + m*lambda*I)^{-1} @ B_t     [smoother at training pts]
                Gamma_t = (B_t + m*lambda*I)^{-1}

                V_res_t = V_{t+1} @ (I - Alpha_t)            [residual coeff at stage t]
                D_pi   += V_res_t * W_pi[t]                  [column-wise IS weighting]

                C_t     = KH_t * KAp_t                       [joint kernel with pi samples]
                V_t     = V_{t+1} @ (C_t @ Gamma_t)^T        [Q^pi_t coeff via policy integration]

            D_pi += V_0                                       [Q^pi_0 base term]

        Cross-fitting: call once per split (L, A, Y are the split's data).
        """
        K = len(L) - 1
        m = len(Y)
        reg = self.reg_lambda
        I_m = np.eye(m)

        # V tracks the coefficient matrix V_{t+1} for pi and pi' simultaneously
        V_pi  = I_m.copy()   # V_{K+1} = I
        V_pip = I_m.copy()

        D_pi  = np.zeros((m, m))
        D_pip = np.zeros((m, m))

        for t in range(K, -1, -1):
            gamma_h = 1.0 / median_bandwidth(L[t])
            gamma_a = 1.0 / median_bandwidth(A[t].reshape(-1, 1))

            KH  = build_kernel_matrix(L[t], metric=self.state_kernel, gamma=gamma_h)                # (m, m)
            KA  = build_kernel_matrix(A[t].reshape(-1, 1), metric=self.state_kernel, gamma=gamma_a) # (m, m)
            B   = KH * KA                                                                # Hadamard
            reg_B = B + m * reg * I_m

            Alpha   = np.linalg.solve(reg_B, B)                          # (m, m) smoother
            Im_A    = I_m - Alpha                                         # (m, m)

            # Residual coefficient: V_{t+1} @ (I - Alpha)
            D_pi  += (V_pi  @ Im_A) * W_pi[t][np.newaxis, :]             # column-wise scale
            D_pip += (V_pip @ Im_A) * W_pi_prime[t][np.newaxis, :]

            # Policy integration: V_t = V_{t+1} @ (C_t @ Gamma_t)^T
            # C @ Gamma = C @ (B + m*lambda*I)^{-1}
            # Solved as: solve(reg_B, C.T).T  (exploits symmetry of reg_B)
            KAp  = build_cross_kernel_matrix(
                pi_samples[t].reshape(-1, 1), A[t].reshape(-1, 1), metric=self.state_kernel, gamma=gamma_a
            )                                                                             # (m, m)
            KApp = build_cross_kernel_matrix(
                pi_prime_samples[t].reshape(-1, 1), A[t].reshape(-1, 1), metric=self.state_kernel, gamma=gamma_a
            )                                                                             # (m, m)

            C_pi  = KH * KAp
            C_pip = KH * KApp

            # CG = C @ Gamma_t, then V_t = V_{t+1} @ CG^T
            CG_pi  = np.linalg.solve(reg_B, C_pi.T).T                    # (m, m)
            CG_pip = np.linalg.solve(reg_B, C_pip.T).T                   # (m, m)

            V_pi  = V_pi  @ CG_pi.T
            V_pip = V_pip @ CG_pip.T

        # Q^pi_0 base term: V after the loop = V^pi_0
        D_pi  += V_pi
        D_pip += V_pip

        return D_pi - D_pip                                               # (m, m)

    # ------------------------------------------------------------------
    # Test variants
    # ------------------------------------------------------------------

    def _test_cross_fit(self, data: LongitudinalOPEData) -> dict:
        """Single n//2 split, uniform across all stages (matches DRKPT._test_cross_fit)."""
        n = len(data.Y)
        m = n // 2

        def _split(sl):
            return (
                [lt[sl] for lt in data.L],
                [at[sl] for at in data.A],
                data.Y[sl],
                [ps[sl] for ps in data.pi_samples],
                [ps[sl] for ps in data.pi_prime_samples],
                data.W_pi[:, sl],
                data.W_pi_prime[:, sl],
            )

        L1, A1, Y1, ps1, pps1, W1, Wp1 = _split(slice(None, m))
        L2, A2, Y2, ps2, pps2, W2, Wp2 = _split(slice(m, None))

        # Fit on I1, evaluate at I1; fit on I2, evaluate at I2 (same as DRKPT)
        D_left  = self._compute_dr_coeff(L1, A1, Y1, ps1, pps1, W1, Wp1)   # (m, m)
        D_right = self._compute_dr_coeff(L2, A2, Y2, ps2, pps2, W2, Wp2)   # (m, m)

        # Cross outcome kernel: k_Y(Y_i, Y_j) for i in I1, j in I2
        Y1_k = Y1 if Y1.ndim > 1 else Y1.reshape(-1, 1)
        Y2_k = Y2 if Y2.ndim > 1 else Y2.reshape(-1, 1)
        KY_cross = build_cross_kernel_matrix(Y1_k, Y2_k, metric="rbf")        # (m, m)

        # prod[i, j] = <Phi_{pi,pi'}(O_i), Phi_{pi,pi'}(O_j)>_{H_Y}
        # Identical structure to dr_kpt.py: left.T @ KY @ right
        prod = D_left.T @ KY_cross @ D_right                                  # (m, m)
        return cross_ustat(prod)

    def _test_full(self, data: LongitudinalOPEData) -> dict:
        """No cross-fitting: fit and evaluate on the full dataset (ablation only)."""
        D = self._compute_dr_coeff(
            data.L, data.A, data.Y,
            data.pi_samples, data.pi_prime_samples,
            data.W_pi, data.W_pi_prime,
        )
        Y_k = data.Y if data.Y.ndim > 1 else data.Y.reshape(-1, 1)
        KY = build_kernel_matrix(Y_k, metric="rbf")
        prod = D.T @ KY @ D
        return cross_ustat(prod)
