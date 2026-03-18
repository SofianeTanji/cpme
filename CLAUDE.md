# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run a Python script
uv run python src/runtime_tables.py

# Run a one-off snippet (always prepend sys.path when importing from src/)
uv run python -c "import sys; sys.path.insert(0, 'src'); from kpt import KPT; ..."

# Open the experiment notebook
uv run jupyter notebook src/experiments.ipynb
```

There is no test suite. Verification is done by running `src/runtime_tables.py` (loads CSVs, writes LaTeX tables) and by running cells in `src/experiments.ipynb`.

## Architecture

This codebase implements **Counterfactual Policy Mean Embeddings (CPME)** — a framework for off-policy evaluation (OPE) and testing whether two policies induce different outcome distributions, using logged bandit data.

### Module dependency order

```
core.py  ←  policies.py  ←  environment.py  ←  datasets.py
   ↑              ↑
kernels.py   kpt.py / dr_kpt.py
```

- **`core.py`** — shared ABCs (`Policy`, `PolicyTest`) and the `OPEData` dataclass. Nothing imports from other project modules.
- **`kernels.py`** — median bandwidth heuristic and kernel matrix builders. Auto-applies the median heuristic for RBF when `gamma=None`; does **not** pass `gamma` for non-RBF kernels (linear kernel rejects it).
- **`policies.py`** — `GaussianPolicy`, `MixturePolicy(policy1, policy2)`, `EstimatedLoggingPolicy`. All implement `Policy`.
- **`environment.py`** — `make_scenario(scenario_id, d, seed)` returns `(X, policy_logging, policy_pi, policy_pi_prime, beta, treatment_effect)` for scenarios I–IV. `generate_ope_data(...)` returns an `OPEData` dataclass (fields: `X, T, Y, w_pi, w_pi_prime, pi_samples, pi_prime_samples`). Importance weights `w_pi` and `w_pi_prime` are shape `(n, 1)`.
- **`datasets.py`** — `Dataset` ABC with `prepare_ope_data(policy_pi, policy_pi_prime) -> OPEData`. `SyntheticDataset(scenario_id, ns, d, seed)` wraps `make_scenario`/`generate_ope_data`; exposes `.policy_pi` and `.policy_pi_prime` for the canonical scenario policies. `RealDataset(X, T, Y, logging_propensities=None)` loads pre-collected data; estimates logging propensities via `EstimatedLoggingPolicy` if not provided. `RealDataset.from_csv(path, x_cols, t_col, y_col, propensity_col)` for CSV loading.
- **`kpt.py`** — `KPT(kernel_function, gamma, iterations, random_state)` implements the kernel policy test. `.test(data)` returns `{"stat": float, "null": ndarray, "pval": float}`. Null distribution uses permutation of importance weights (same index applied to both `w_pi` and `w_pi_prime`).
- **`dr_kpt.py`** — `DRKPT(kernel_function, reg_lambda, cross_fit)` implements the doubly-robust variant. `.test(data)` returns `{"stat": float, "pval": float}` (z-statistic, no permutation null). `DRKPT.tune_reg_lambda(X, T, Y)` runs CV-KRR to select `reg_lambda`. When `cross_fit=True`, bandwidth is estimated on the second half of data (`slice(N//2, None)`) while full gram matrices span all N rows.
- **`runtime_tables.py`** — `load_results(results_folder)` returns a dict keyed by CSV file path. `build_scenario_table(scenario, sample_size_list, d_results)` now takes `d_results` as an explicit argument (no global). `_result_path()` is the single source of truth for CSV file naming: `ns{ns}_scenario{scenario}_{method}.csv`.
- **`experiments.ipynb`** — orchestrates experiments via `run_ope_tests(...)`. Results are written to `src/results/`. Plots go to `src/plots/`. Tables go to `src/tables/`.

### Statistical methods

| Class | Method name in results CSVs | p-value type |
|---|---|---|
| `KPT(kernel_function="linear")` | `PE-linear` | permutation |
| `KPT(kernel_function="rbf")` | `KPE` | permutation |
| `DRKPT(cross_fit=False)` | `DR` | normal CDF |
| `DRKPT(cross_fit=True)` | `DR-CF` | normal CDF |

### Scenarios

- **I**: Null (π = π′), sample sizes up to 1000
- **II**: Mean shift in π′
- **III**: π′ is a bimodal mixture policy
- **IV**: π′ is a half-shifted mixture policy
- Scenarios II–IV use sample sizes 100–400 (step 50)
