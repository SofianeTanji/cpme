# Nested Orthogonal Kernel Methods for Longitudinal Distributional Causal Effects

## Starting point: non-nested CPME (Zenati et al., NeurIPS 2025)

### Setup

The current codebase implements the framework from *Doubly-Robust Estimation of Counterfactual
Policy Mean Embeddings* (Zenati, Bozkurt, Gretton, NeurIPS 2025). The setting is a single-stage
contextual bandit. Data are i.i.d. triples `(x_i, a_i, y_i) ~ P_0 = P_{Y|X,A} × g_0 × P_X`.

The object of interest is the **Counterfactual Policy Mean Embedding (CPME)**:

```
χ(π) = E_{P_π}[φ_Y(Y(a))]  ∈  H_Y
```

This is the kernel mean embedding of the counterfactual outcome distribution `ν(π)`. Representing
`ν(π)` as an RKHS element enables hypothesis testing, sampling, and distributional comparisons
without restricting to scalar moments.

### Identification and plug-in estimator

Under selection on observables (`Y(a) ⊥ A | X`), the CPME is identified as:

```
χ(π) = E_{π × P_X}[ μ_{Y|A,X}(a, x) ]
```

where `μ_{Y|A,X}(a, x) = E_{P_{Y|X,A}}[φ_Y(Y) | A=a, X=x]` is the conditional mean embedding
evaluated at `(a, x)`. This decoupling — **one operator `C_{Y|A,X}` acting on one embedding
`μ_π`** — is what makes the setting "non-nested." The plug-in estimator fits `Ĉ_{Y|A,X}` via
KRR on `(a_i, x_i) → φ_Y(y_i)` and estimates `μ_π` empirically.

### Efficient influence function and DR estimator

The Hilbert-space-valued EIF of `χ(π)` is:

```
ψ^π(y, a, x) = [π(a|x) / g_0(a|x)] { φ_Y(y) − μ_{Y|A,X}(a, x) }
               + β_π(x) − χ(π)
```

where `β_π(x) = ∫ μ_{Y|A,X}(a', x) π(da' | x)`. The one-step DR estimator is:

```
χ̂_dr(π) = (1/n) Σ_i { w_π(a_i, x_i) [φ_Y(y_i) − μ̂_{Y|A,X}(a_i, x_i)] + β̂_π(x_i) }
```

Doubly robust: consistent if either `ĝ_0` or `μ̂_{Y|A,X}` is correctly specified.

### DR kernel policy test (DR-KPT)

To test `H_0: ν(π) = ν(π')`, the paper forms the difference of EIFs:

```
φ_{π,π'}(y, a, x) = [π(a|x)/g_0(a|x) − π'(a|x)/g_0(a|x)] { φ_Y(y) − μ_{Y|A,X}(a, x) }
                    + β_π(x) − β_{π'}(x)
```

The test statistic is a cross U-statistic with nuisance estimated on separate halves:

```
T^†_{π,π'} = √m  f̄^†_{π,π'} / S^†_{π,π'}
```

where `f^†(i,j) = ⟨φ̂_{π,π'}(y_i, a_i, x_i),  φ̃_{π,π'}(y_j, a_j, x_j)⟩_{H_Y}`. This
statistic is asymptotically `N(0,1)` under `H_0`, avoiding permutation.

In code: `DRKPT._test_full` and `DRKPT._test_cross_fit` in `src/dr_kpt.py`. The DR term
`(w_pi_prime − w_pi)(I − mu_logging) + mu_pi_prime − mu_pi` is the matrix realization of
`φ_{π,π'}`.

---

## The gap: single-stage identification fails under treatment-confounder feedback

The identification assumption `Y(a) ⊥ A | X` requires all confounders captured by `X` at a
single cross-section. In longitudinal settings, `A_t` can affect `L_{t+1}`, which confounds
`A_{t+1}`. This is treatment-confounder feedback and is not handled by single-stage
exchangeability.

With `K = 1`, the joint distribution is:

```
P_0 = P_{Y | A_0, A_1, L_0, L_1}  ×  g_{0,0}(A_0|L_0)  ×  P_{L_1|A_0,L_0}  ×  g_{1,0}(A_1|H_1)  ×  P_{L_0}
```

`L_1` depends on `A_0`, so it is a time-varying confounder for `A_1`. Identifying the
counterfactual distribution under `π` requires integrating out `L_1` under `P(L_1|A_0,L_0)` —
a second regression problem — before integrating out `L_0`. This cannot be collapsed into a
single operator `C_{Y|A,X} μ_π`.

Applying the non-nested estimator in this setting yields a biased plug-in and a DR correction
that does not remove all first-order bias, invalidating the asymptotic normality of the test.

---

## The extension: nested CPME for longitudinal policies

### Longitudinal setting

Fix a finite horizon `K ≥ 0`. For each unit `i = 1, ..., n`, observe a longitudinal trajectory:

```
O_i = (L_{0,i}, A_{0,i}, L_{1,i}, A_{1,i}, ..., L_{K,i}, A_{K,i}, Y_i)
```

where `L_t ∈ L_t` are time-`t` covariates, `A_t ∈ A_t` is the treatment at time `t`, and
`Y ∈ Y` is the terminal outcome observed after time `K`. The **history available prior to the
assignment of `A_t`** is:

```
H_t := (L_0, A_0, L_1, A_1, ..., L_t)
```

Note: `H_t` includes `L_t` but **not** `A_t`. The statistical model factorizes as:

```
p_0(o) = p_0(l_0)  ∏_{t=0}^K g_{t,0}(a_t | h_t)  ∏_{t=1}^K p_0(l_t | h_{t-1}, a_{t-1})  p_0(y | h_K, a_K)
```

where `g_{t,0}(· | h_t)` is the (unknown) logging policy at time `t`. No parametric assumptions
are imposed on any component.

A target policy is `π = (π_0, ..., π_K)` where each `π_t(· | h_t)` is a probability
distribution on `A_t`. Static, dynamic, and stochastic regimes are all encompassed. The
**nested CPME** is:

```
μ^π := Ψ_π(P_0) = E_{P^π}[φ(Y)]  ∈  H_Y
```

### Identification via nested g-computation

Under sequential ignorability (`Y^π ⊥⊥ A_t | H_t` for all `t`), consistency, and positivity
(Assumptions 2.1–2.3 of the draft), `μ^π` is nonparametrically identified via the following
backward recursion (Proposition 2.4 of the draft).

Define the sequence of `H_Y`-valued functions `(Q^π_t)_{t=0}^{K+1}` recursively:

```
Q^π_{K+1}(H_{K+1})    :=  φ(Y)                                             ← terminal condition

Q̄^π_{t+1}(h_t, a_t)  :=  E_{P_0}[ Q^π_{t+1}(H_{t+1}) | H_t = h_t, A_t = a_t ]  ← Bochner conditional expectation

Q^π_t(h_t)            :=  ∫ Q̄^π_{t+1}(h_t, a) π_t(da | h_t)              ← integrate under π_t
```

Then:

```
μ^π = E_{P_0}[ Q^π_0(H_0) ]
```

The alternating structure — conditional expectation under `P_0`, then integration under `π_t`
— respects causal ordering and accounts for treatment-confounder feedback through `H_t`.

### Canonical gradient and stagewise Riesz representation

**Theorem 3.1** of the draft establishes that `Ψ_π` is pathwise differentiable and gives the
canonical gradient explicitly:

```
Φ_π(O) = Σ_{t=0}^K  W^π_t(H_t, A_t)  [ Q^π_{t+1}(H_{t+1}) − Q̄^π_{t+1}(H_t, A_t) ]
          +  Q^π_0(H_0)  −  μ^π
```

where the cumulative importance weight up to stage `t` is:

```
W^π_t(H_t, A_t) = ∏_{s=0}^t  π_s(A_s | H_s) / g_{s,0}(A_s | H_s)
```

Each summand is a weighted `H_Y`-valued residual: `Q^π_{t+1}(H_{t+1})` is the next-stage
function evaluated at the *realized* next history, and `Q̄^π_{t+1}(H_t, A_t)` is its
conditional expectation. The stage-`t` residual is the difference.

**Stagewise Riesz functionals.** For each stage `t`, define the bounded linear functional on
scalar functions `f` over `H_{t+1}`:

```
ℓ^π_t(f) := E_{P_0}[ W^π_t(H_t, A_t)  (f(H_{t+1}) − E_{P_0}[f(H_{t+1}) | H_t, A_t]) ]
```

By the Riesz representation theorem, there exists a unique scalar function `r^π_t ∈ H_{t+1}`
satisfying `ℓ^π_t(f) = ⟨r^π_t, f⟩_{H_{t+1}}` for all `f`. These are the **stagewise Riesz
representers**. The target embedding decomposes as:

```
μ^π = Σ_{t=0}^K  E_{P_0}[ r^π_t(H_{t+1})  Q^π_{t+1}(H_{t+1}) ]
```

Estimating `(r^π_t)_{t=0}^K` jointly would yield a single coupled inverse problem over
`H_1 ⊕ ... ⊕ H_{K+1}`. The decomposition above gives `K+1` **decoupled** inverse problems,
each on its own history space with stage-specific regularization.

### On propensity estimation

The `Q^π_t` backward recursion (conditional Bochner expectations under `P_0`) requires no
propensity estimation — it uses kernel regression of `H_Y`-valued quantities on `(H_t, A_t)`.

Propensity scores `g_{t,0}(A_t | H_t)` appear only in the IS weights `W^π_t`. These are either:

- **Known**: in randomized experiments or when the logging policy is a known stochastic policy.
  Weights are computed exactly.
- **Estimated**: when `g_{t,0}` is unknown, requiring `K+1` separate propensity models, each
  taking `H_t` as input.

The implementation should support both modes, mirroring how the current codebase handles
`EstimatedLoggingPolicy` vs known propensities.

### Estimation

**One-step estimator.** Given estimates `(r̂^π_t, Q̂^π_{t+1})_{t=0}^K`, the one-step embedding
estimator is:

```
μ̂^π = Σ_{t=0}^K  (1/n) Σ_{i=1}^n  r̂^π_t(H_{t+1,i})  Q̂^π_{t+1}(H_{t+1,i})
```

a scalar–H_Y inner product at each observation, summed over stages and units.

**Stagewise Riesz regression.** For each stage `t`, `r̂^π_t` is obtained by solving:

```
r̂^π_t = argmin_{r ∈ H_{t+1}}  (1/n) Σ_i  ( r(H_{t+1,i}) − ζ̂_{t,i} )²  +  λ_t ||r||²_{H_{t+1}}
```

where the regression target is:

```
ζ̂_{t,i} = W^π_t(H_{t,i}, A_{t,i})  −  T̂^π_t W^π_t(H_{t,i}, A_{t,i})
```

i.e., the IS weight at stage `t` minus its estimated conditional expectation given `(H_t, A_t)`.

**Nested Q-function estimation.** The `Q^π_t` functions are estimated by the backward recursion:

```
Q̂^π_{K+1}(H_{K+1,i})  :=  φ(Y_i)

Q̂^π_{t+1}(h_t, a_t)   :=  Ê[ Q̂^π_{t+1}(H_{t+1}) | H_t = h_t, A_t = a_t ]   (H_Y-valued KRR)

Q̃^π_t(h_t)            :=  ∫ Q̂^π_{t+1}(h_t, a) π_t(da | h_t)
```

Each conditional expectation is estimated as an `H_Y`-valued regression (conditional mean
embedding) of `Q̂^π_{t+1}(H_{t+1})` on `(H_t, A_t)`.

The policy integration `Q̃^π_t` follows exactly the same pattern as `KT_pi` in the current
codebase: the kernel mean embedding of `π_t` is evaluated at training points and passed as
an argument to the KRR solve. This is a kernel evaluation — not sampling from the
counterfactual outcome distribution (which is a separate capability, Algorithm 2 of Zenati
et al., not needed for testing). The interface is agnostic to how the policy's kernel mean
embedding is computed.

**Cross-fitting.** A single `m = ⌊n/2⌋` split is used: fit all nuisance components
`(r̂^π_t, Q̂^π_{t+1})` on `I_1 = {1, ..., m}`, evaluate `Φ̂_{π,π'}` on
`I_2 = {m+1, ..., n}` (and symmetrize by swapping roles if desired). The same split applies
uniformly to all `K+1` stages — no stagewise partitioning needed. This is consistent with
Section 4.2 of the draft and avoids multiple levels of sample splitting.

---

## Research questions

**Q1 — Two-sample test.** Given logged trajectory data under `g_{t,0}`, test whether two
longitudinal policies `π` and `π'` induce different counterfactual outcome distributions:

```
H_0: μ^π = μ^{π'}   vs.   H_1: μ^π ≠ μ^{π'}
```

Since `k_Y` is characteristic, `μ^π − μ^{π'} = E_{P_0}[Φ_{π,π'}(O)]` where
`Φ_{π,π'}(O) := Φ_π(O) − Φ_{π'}(O)`. The cross U-statistic is:

```
f_i    =  (1/|I_2|) Σ_{j∈I_2}  ⟨Φ̂_{π,π'}(O_i),  Φ̂_{π,π'}(O_j)⟩_{H_Y}

T^†_n  =  √|I_1|  f̄ / S
```

Under `H_0`, cross-fitting, the rate condition `Σ_t ||r̂^π_t − r^π_t||_{L^2} ||Q̂^π_{t+1} −
Q^π_{t+1}||_{L^2(P_0;H_Y)} = o_P(n^{-1/2})`, and non-degeneracy, Theorem 5.1 of the draft
gives `T^†_n →^d N(0,1)`.

**Q2 — Localization.** Section 5.2 of the draft is blank. This is genuinely open, not yet
formulated.

---

## Implementation design questions

### 1. Kernel over history H_t

`H_t = (L_0, A_0, ..., L_t)` grows with `t`. The natural choice for the KRR at stage `t` is a
**tensor product kernel** over the components of `H_t`:

```
k_{H_t}(h, h') = k_{L_0}(l_0, l_0') · k_{A_0}(a_0, a_0') · ... · k_{L_t}(l_t, l_t')
```

An alternative is to restrict to only the most recent `(L_t, A_{t-1})` (Markov assumption),
which reduces the kernel matrix size but restricts generality. **Decision needed**: full history
vs Markov kernel at each stage.

### 2. Cross-fit design

Resolved by the draft (Section 4.2): **single `n/2` split**, uniform across all stages. No
stagewise partitioning. Symmetrization (swapping split roles) is optional.

### 3. Synthetic scenarios for longitudinal testing

Minimum viable set for experiments:

- **Scenario I (Null)**: `π = π'`, treatment-confounder feedback present, `ν(π) = ν(π')`
- **Scenario II (Mean shift)**: `π'` has a shifted mean at each stage
- **Scenario III (Mixture)**: `π'` is a bimodal mixture at one or more stages
- **Scenario IV (Feedback-only shift)**: `π` and `π'` agree marginally at each stage but differ
  in how they respond to time-varying confounders `L_t` — a distributional shift that is
  invisible to non-nested methods and only detectable via the nested structure

Scenario IV is the critical one: it directly tests the added value of the nested approach over
a naive single-stage estimator.

---

## Datasets

### US Job Corps (Singh et al., 2025)

Singh et al. cleaned and published the Job Corps longitudinal dataset as a benchmark:

- `L_0 ∈ R^{40}`: baseline covariates
- `A_0 ∈ R`: class hours in year 1 (continuous treatment)
- `L_1 ∈ R^{30}`: time-varying covariates at end of year 1 (post-treatment confounders)
- `A_1 ∈ R`: class hours in year 2
- `Y ∈ R`: outcome (employment or arrests at year 4)
- `n ≈ 2913–3141`, `K = 1`

`K = 1` with continuous treatments and treatment-confounder feedback (`L_1` affected by `A_0`).
A natural pair `(π, π')` can be two dosage regimes; the test asks whether they induce different
outcome distributions. Replication package: DOI 10.3150/24-BEJ1836SUPPB.

### Finding additional datasets — prompt for LLM search

```
I am working on a method for testing whether two longitudinal causal policies induce
different outcome distributions. I need real-world datasets satisfying these criteria:

1. Longitudinal structure: each observation is a trajectory with at least K=1 time steps
   (i.e., at least two treatment stages), with covariates, treatments (ideally continuous),
   and a terminal outcome.

2. Treatment-confounder feedback: intermediate covariates measured between treatment stages
   that are causally affected by prior treatments and also confound later treatments.
   This is the key structural requirement — datasets where all confounders are baseline
   covariates do NOT qualify.

3. Used in recent causal inference research (2018–2025), particularly in papers on:
   - Time-varying treatment effects
   - Marginal structural models
   - Sequential doubly robust estimation
   - G-computation or g-formula estimation
   - Dynamic treatment regimes (DTRs)
   - Nested structural mean models

4. Publicly available (or available on request), with at least n=500 observations.

5. Continuous or near-continuous treatments preferred over binary, since the method is
   designed for continuous treatment spaces.

For each dataset, provide:
- Dataset name and source
- The treatment(s), time-varying covariates, and outcome
- Approximate sample size and horizon K
- The paper(s) that used it
- Whether it is publicly available and where

Good candidate sources: NHANES, MIMIC-III/IV, Job Corps (already have this one),
ACTG clinical trials, UK Biobank longitudinal modules, SHARE (Survey of Health Ageing
and Retirement in Europe), Add Health, National Longitudinal Survey of Youth (NLSY).
```

---

## Summary of differences from the current codebase

| Aspect | Current (`dr_kpt.py`) | This extension |
|---|---|---|
| Data | `(X, A, Y)` i.i.d. | `(L_{0:K}, A_{0:K}, Y)` trajectories |
| History | `X` (single cross-section) | `H_t = (L_0, A_0, ..., L_t)`, prior to `A_t` |
| Identification | `Y(a) ⊥ A \| X` | Sequential ignorability `Y^π ⊥⊥ A_t \| H_t` |
| CPME structure | `C_{Y\|A,X} μ_π` — one operator | `K+1`-step backward recursion over `H_Y`-valued `Q^π_t` |
| Logging policy | Single `g_0(a\|x)` | Stagewise `g_{t,0}(a_t \| h_t)` for `t = 0, ..., K` |
| IS weights | `w_π = π(a\|x)/g_0(a\|x)` | `W^π_t = ∏_{s≤t} π_s(A_s\|H_s)/g_{s,0}(A_s\|H_s)` |
| Nuisance 1 | 1 KRR (`_compute_dr_term`) | `K+1` `H_Y`-valued KRRs for `Q̂^π_t`, backward |
| Nuisance 2 | Implicit via propensity ratio | `K+1` scalar Riesz regressions for `r̂^π_t` |
| Riesz target | — | `ζ̂_{t,i} = W^π_t − T̂^π_t W^π_t` |
| One-step estimator | Matrix DR term | `Σ_t (1/n) Σ_i r̂^π_t(H_{t+1,i}) Q̂^π_{t+1}(H_{t+1,i})` |
| Cross-fitting | Single `n/2` split | Same — single `n/2` split, uniform across stages |
| Test statistic | Cross U-stat on `φ_{π,π'}` | Cross U-stat on `Φ_{π,π'} = Φ_π − Φ_{π'}` |
| Q2 | Not addressed | Not yet formulated (blank in draft) |

The current `OPEData` dataclass, `DRKPT` class, and `_compute_dr_term` method are all specific
to `K = 0`. The longitudinal extension requires new data structures (trajectory arrays indexed
by stage), a backward induction loop over `H_Y`-valued regressions, `K+1` stagewise Riesz
regressions, and a new test class assembling the stagewise terms into the cross U-statistic.
