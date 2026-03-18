# Implementation plan: nested CPME for longitudinal testing

## Goal

Extend the existing K=0 (single-stage) CPME codebase to handle K≥1 longitudinal trajectories
with treatment-confounder feedback. The new test class (`NestedDRKPT`) produces an asymptotically
N(0,1) statistic and is exercised in four synthetic scenarios (including Scenario IV, which is
invisible to the non-nested method).

---

## What is reused without modification

| Component | Location | Why it transfers |
|---|---|---|
| `build_kernel_matrix`, `build_cross_kernel_matrix`, `median_bandwidth` | `kernels.py` | Operate on arbitrary input arrays; agnostic to what they represent |
| Cross U-statistic formula (`prod → U → stat`) | `dr_kpt.py:_test_cross_fit` L126-130 | Identical computation; copy into `NestedDRKPT._cross_ustat` |
| `GaussianPolicy.get_propensities(L_t, A_t)` | `policies.py` | Computes g_{t,0}(A_t\|L_t) under Markov assumption if passed L_t |
| `MixturePolicy` | `policies.py` | Used for Scenarios III and IV stage-level policies |

---

## Files to create

### Step 1 — `src/core.py`: add `LongitudinalOPEData`

```python
@dataclass
class LongitudinalOPEData:
    L          : list[np.ndarray]   # L[t] shape (n, d_t),  t = 0..K
    A          : list[np.ndarray]   # A[t] shape (n,),       t = 0..K
    Y          : np.ndarray         # shape (n,)
    W_pi       : np.ndarray         # shape (K+1, n)  — W^π_t for each t
    W_pi_prime : np.ndarray         # shape (K+1, n)
    pi_samples       : list[np.ndarray]   # pi_samples[t] shape (n,)  — ã_i ~ π_t(·|L_{t,i})
    pi_prime_samples : list[np.ndarray]
```

`W_pi[t, i]` is the cumulative IS weight through stage t:
`W^π_t = ∏_{s=0}^t π_s(A_{s,i}|L_{s,i}) / g_{s,0}(A_{s,i}|L_{s,i})`

---

### Step 2 — `src/longitudinal_environment.py`

**`make_longitudinal_scenario(scenario_id, K=1, d=5, seed=None)`**

Returns `(logging_policies, pi_policies, pi_prime_policies, beta, treatment_effect)` where each
argument is a list of length K+1 (one policy per stage). All policies are `GaussianPolicy` or
`MixturePolicy` instances acting on `L_t` under the Markov assumption.

Data-generating process (K=1, d=5):
```
L_0   ~ N(0, I_d)
A_0   ~ π_{log,0}(·|L_0)   = GaussianPolicy(w_base)(L_0)
L_1   = L_0 @ B + A_0 * c + ε_1    # treatment-confounder feedback; ε_1 ~ N(0, 0.5 I_d)
A_1   ~ π_{log,1}(·|L_1)
Y     = L_K @ beta + treatment_effect * A_K + noise
```

Where `B` is a d×d mixing matrix (e.g. `0.5 * I_d`) and `c` is a d-vector (`0.3 * ones(d)`).

Scenarios:
- **I (Null)**: π_t = π'_t at every stage; feedback present
- **II (Mean shift)**: π'_t has weight vector `w_base + 2*ones(d)` at each stage
- **III (Mixture)**: π'_t is `MixturePolicy` at each stage (bimodal)
- **IV (Feedback-only shift)**: π_t and π'_t have the same marginal over A_t given L_t, but
  differ in how they respond to L_t's temporal structure. Concretely: same w_base, but π'_1
  uses `L_0` (lagged covariate) instead of `L_1` as the conditioning variable. This makes π
  and π' indistinguishable to any non-nested test but detectable by the nested test.

**`generate_longitudinal_ope_data(L0, logging_policies, pi_policies, pi_prime_policies, beta, treatment_effect, K=1, noise_std=0.1) -> LongitudinalOPEData`**

Loop over stages 0..K:
1. Sample A_t ~ logging_policies[t](·|L_t)
2. Compute L_{t+1} from feedback model
3. Compute cumulative IS weights W_pi[t] and W_pi_prime[t]
4. Sample ã_t ~ pi_policies[t](·|L_t) for policy integration (pi_samples[t])
5. Generate Y from L_K, A_K

Returns `LongitudinalOPEData`.

---

### Step 3 — `src/nested_dr_kpt.py`

**`NestedDRKPT(reg_lambda=1e-2, mc_samples=1, cross_fit=True)`**

#### `test(data: LongitudinalOPEData) -> dict`

Dispatches to `_test_cross_fit` (primary) or `_test_full` (debug/ablation).

#### Internal: backward KRR loop

For a given data split of size m (either full n or n//2):

```
M_{K+1} = build_kernel_matrix(Y, metric="rbf")   # shape (m, m)

for t = K, K-1, ..., 0:
    KH_t  = build_kernel_matrix(L[t], metric="rbf")           # (m, m)
    KA_t  = build_kernel_matrix(A[t][:,None], metric="rbf")   # (m, m)
    KA_pi = build_cross_kernel_matrix(
                pi_samples[t][:,None], A[t][:,None], metric="rbf"
            )                                                   # (m, m)

    B_t = KH_t * KA_t                                          # Hadamard, (m, m)
    Alpha_t = solve(B_t + m*reg_lambda*I, B_t)                 # (m, m)

    # Gram of Q̄^π_{t+1} at training points (for DR residual)
    G_bar_{t+1} = Alpha_t @ M_{t+1} @ Alpha_t                  # (m, m)

    # Gram of Q^π_t at training points (via policy integration)
    C_t = KH_t * KA_pi                                         # (m, m); uses π_t samples
    M_t = C_t @ Alpha_t @ M_{t+1} @ Alpha_t @ C_t.T           # (m, m)

    store (Alpha_t, G_bar_{t+1}, M_t, B_t) for this stage
```

The DR residual Gram matrix at each stage t (needed for the EIF inner products):
```
R_t = M_{t+1} - G_bar_{t+1} - G_bar_{t+1}.T + G_bar_{t+1}
    = (I - Alpha_t) @ M_{t+1} @ (I - Alpha_t)               # (m, m)
```

This is the Gram matrix of `[Q^π_{t+1}(H_{t+1,i}) - Q̄^π_{t+1}(H_{t,i}, A_{t,i})]`.

#### Inner product computation

For the cross U-statistic, we need:
```
f(i, j) = <Φ_{π,π'}(O_i), Φ_{π,π'}(O_j)>_{H_Y}
```

This decomposes into:
```
f(i,j) = Σ_t (W_pi[t,i] - W_pi_prime[t,i]) * (W_pi[t,j] - W_pi_prime[t,j]) * R_t_pi[i,j]
          - Σ_t ... [analogous terms for π' residuals]
          + M_0_pi[i,j] - M_0_piprime[i,j] - ...
```

where R_t_pi, R_t_piprime are the DR residual Gram matrices for π and π' respectively, and
M_0_pi, M_0_piprime are the Q^π_0 and Q^{π'}_0 Gram matrices.

Full assembled prod matrix (m×m):
```
prod = Σ_t outer(W_pi[t] - W_pi_prime[t]) * R_t
       + M_0_pi + M_0_piprime - M_0_cross - M_0_cross.T
```

where `outer(w)[i,j] = w[i]*w[j]` and `M_0_cross` is the cross Gram between Q^π_0 and Q^{π'}_0.

#### Cross U-statistic (reused)

```python
def _cross_ustat(self, prod: np.ndarray) -> dict:
    U = prod.mean(axis=1)
    stat = float(np.sqrt(len(U)) * U.mean() / U.std())
    return {"stat": stat, "pval": float(1 - norm.cdf(stat))}
```

#### `_test_cross_fit`

Single n//2 split, uniform across all stages:
- I1 = data[:n//2], I2 = data[n//2:]
- Fit backward KRR on I1; evaluate prod matrix on I2 (using cross kernel I2 vs I1 for KH, KA)
- Fit backward KRR on I2; evaluate prod matrix on I1
- Symmetrize: average the two statistics (or use only one split)

#### `tune_reg_lambda(L_list, A_list, Y, reg_grid=None) -> float`

CV-KRR on concatenated `(L_K, A_K)` → Y. Identical structure to `DRKPT.tune_reg_lambda`,
reusable pattern.

---

### Step 4 — `src/longitudinal_datasets.py`

```python
class LongitudinalSyntheticDataset:
    def __init__(self, scenario_id, K=1, ns=200, d=5, seed=None):
        ...
    def prepare_ope_data(self) -> LongitudinalOPEData:
        ...
```

Thin wrapper around `make_longitudinal_scenario` + `generate_longitudinal_ope_data`.
Analogous to `SyntheticDataset` in `datasets.py`.

---

### Step 5 — `tests/test_nested_dr_kpt.py`

Minimal smoke tests (no simulation, small n):

```python
def _make_longitudinal_data(n=60, K=1, seed=0) -> LongitudinalOPEData: ...

def test_result_keys(): ...           # {"stat", "pval"} present
def test_pval_in_unit_interval(): ... # 0 ≤ pval ≤ 1
def test_stat_is_scalar(): ...        # isinstance float
def test_null_pval_not_tiny(): ...    # W_pi == W_pi_prime → pval not near 0
def test_k0_matches_drkpt(): ...      # K=0 with flat history matches DRKPT output
```

---

### Step 6 — `src/longitudinal_experiments.ipynb`

Cells mirror the structure of `experiments.ipynb`:

1. **Imports** — `NestedDRKPT`, `LongitudinalSyntheticDataset`, etc.
2. **`run_longitudinal_tests(scenario_list, ns_list, num_experiments, name_folder)`**
   Same loop structure as `run_ope_tests`; saves `ns{ns}_longitudinal_scenario{s}_NestedDR-CF.csv`.
3. **Null diagnostics** (Scenario I): histogram + QQ + false-positive-rate vs n
4. **Power curves** (Scenarios II, III, IV): rejection rate vs n for `NestedDR-CF`
5. **Comparison plot** (Scenario IV only): `NestedDR-CF` vs `DR-CF` (non-nested) — the
   key result showing that the non-nested test is powerless while the nested test is consistent.

---

## Implementation order

```
1. core.py              add LongitudinalOPEData
2. longitudinal_environment.py   scenarios I–IV with feedback
3. nested_dr_kpt.py              NestedDRKPT class
4. tests/test_nested_dr_kpt.py   smoke tests
5. longitudinal_datasets.py      thin dataset wrapper
6. longitudinal_experiments.ipynb  full experiment pipeline
```

Steps 1–4 are the load-bearing pieces. Steps 5–6 are scaffolding around them.

---

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| History kernel | Markov: kernel on L_t only | Keeps matrix size O(n²); full H_t kernel grows with t |
| Cross-fit split | Single n//2, uniform across stages | Matches draft Section 4.2; no stagewise partitioning |
| Policy integration | mc_samples=1: one draw ã_i ~ π_t(·|L_{t,i}) | Matches existing KT_pi pattern; increase mc_samples for sensitivity check |
| Riesz regression | Same reg_lambda as Q regression; no cross-val per stage | Keep implementation simple; tune once on (L_K, A_K) → Y |
| Scenario IV | π'_1 conditions on L_0 instead of L_1 | Marginals match π, but feedback-response differs — non-nested blind |
