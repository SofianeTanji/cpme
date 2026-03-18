import numpy as np
from scipy.stats import norm
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_kernels, pairwise_distances
from sklearn.model_selection import GridSearchCV


def median_bandwidth(Z: np.ndarray) -> float:
    """sigma^2 = median(pairwise euclidean distances)^2. Returns 1.0 if zero."""
    dists = pairwise_distances(Z, metric="euclidean")
    med = np.median(dists)
    return float(med ** 2) if med > 0 else 1.0


def rbf_gamma_from_median(Z: np.ndarray) -> float:
    """Returns 1 / median_bandwidth(Z) for sklearn's RBF gamma."""
    return 1.0 / median_bandwidth(Z)


def build_kernel_matrix(Z, metric="rbf", gamma=None, **kwargs) -> np.ndarray:
    """Builds (n×n) kernel matrix; auto median heuristic when metric='rbf' and gamma=None."""
    if metric == "rbf":
        if gamma is None:
            gamma = rbf_gamma_from_median(Z)
        kwargs["gamma"] = gamma
    return pairwise_kernels(Z, metric=metric, **kwargs)


def build_cross_kernel_matrix(Z1, Z2, metric="rbf", gamma=None, **kwargs) -> np.ndarray:
    """Builds (n×m) cross-kernel matrix K(Z1, Z2); gamma inferred from Z1."""
    if metric == "rbf":
        if gamma is None:
            gamma = rbf_gamma_from_median(Z1)
        kwargs["gamma"] = gamma
    return pairwise_kernels(Z1, Z2, metric=metric, **kwargs)


def tune_reg_lambda(X, T, Y, reg_grid=None, num_cv=3) -> float:
    """Cross-validated KRR on [X, T] -> Y to select reg_lambda."""
    if reg_grid is None:
        reg_grid = [1e1, 1e0, 0.1, 1e-2, 1e-3, 1e-4]
    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        cv=num_cv,
        param_grid={"alpha": reg_grid},
    )
    features = np.concatenate([X, T.reshape(-1, 1)], axis=1)
    kr.fit(features, Y.ravel())
    return kr.best_params_["alpha"]


def cross_ustat(prod: np.ndarray) -> dict:
    """Cross U-statistic: z-test from prod matrix. Returns {"stat": float, "pval": float}."""
    U = prod.mean(axis=1)
    stat = float(np.sqrt(len(U)) * U.mean() / U.std())
    return {"stat": stat, "pval": float(1 - norm.cdf(stat))}
