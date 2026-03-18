# Experiment Design: Doubly Robust Kernel Tests for Nested Counterfactual Mean Embeddings

## 1. Objective
To empirically validate a doubly robust (DR) sequential kernel hypothesis test for longitudinal causal inference using an existing codebase. The experiment uses a semi-synthetic, fully Markovian data generating process (DGP) with continuous treatments and a terminal 5D continuous outcome.

## 2. Data Generating Process (DGP)
We simulate a two-step Markovian trajectory $O = (L_1, A_1, L_2, A_2, Y)$ where all intermediate variables are continuous and properly shaped for `sklearn` compatibility.

### 2.1. Structural Equations & Exact Mappings
Let $n$ be the sample size. For each individual $i \in \{1, \dots, n\}$:
* **Baseline Confounder:** $L_{1i} \sim \mathcal{N}(0, 1)$ *(Implemented with shape `(n, 1)` to prevent sklearn kernel dimension errors)*
* **First Treatment (Behavioral Logging):** $A_{1i} \sim \mathcal{N}(0.3 \cdot L_{1i}, 1)$ 
* **Intermediate Confounder:** $L_{2i} = 0.5 \cdot L_{1i} + 0.5 \cdot A_{1i} + \mathcal{N}(0, 1)$ *(Shape `(n, 1)`)*
  *(Nested structure: $L_2$ mediates the effect of $A_1$)*
* **Second Treatment (Behavioral Logging):** $A_{2i} \sim \mathcal{N}(0.3 \cdot L_{2i}, 1)$

### 2.2. Terminal Structured Outcome ($Y$)
To satisfy the backward Kernel Ridge Regression (KRR) assumption, the outcome $Y$ must be strictly terminal (a function of $L_2, A_2$ only). We map $(L_2, A_2)$ to a 5-dimensional continuous vector:
$$Y_i = \begin{bmatrix} 
L_{2i} \\ 
\sin(L_{2i}) \\ 
A_{2i} \\ 
\cos(A_{2i}) \\ 
L_{2i} \cdot A_{2i}
\end{bmatrix} + \epsilon_i$$
*(where $\epsilon_i \sim \mathcal{N}(0, 0.1 \cdot I_5)$).*

## 3. Hypothesis Test & Target Policies
We evaluate continuous **stochastic shift policies**. To ensure Importance Sampling (IS) weights do not trivially collapse to 1.0 under the null hypothesis, the base evaluation policy intentionally differs from the behavioral logging policy ($0.5$ vs $0.3$ multiplier).

Let $\pi_e$ be the baseline target policy and $\pi_e'$ be a shifted policy controlled by $\Delta$:
* $\pi_{e,1}(A_1 \mid L_1) = \mathcal{N}(0.5 \cdot L_1, 1)$
* $\pi_{e,1}'(A_1 \mid L_1) = \mathcal{N}(0.5 \cdot L_1 + \Delta, 1)$
* $\pi_{e,2}(A_2 \mid L_2) = \mathcal{N}(0.5 \cdot L_2, 1)$
* $\pi_{e,2}'(A_2 \mid L_2) = \mathcal{N}(0.5 \cdot L_2 + \Delta, 1)$

We test the null hypothesis:
$$H_0: \mu_{\pi_e} = \mu_{\pi_e'}$$

**Test Statistic:**
The asymptotic **Z-statistic** returned by `NestedDRKPT`.
*Kernels:* RBF kernels for the treatments $A_t$ and an RBF cross-kernel for the 5D outcome $Y$.

## 4. Evaluation Metrics
The experiment evaluates validity and power over $N_{sim} = 100$ Monte Carlo runs.

### 4.1. Type I Error Control (Validity under $H_0$)
* **Setup:** Set the policy shift $\Delta = 0$. Thus, $\pi_e = \pi_e'$, making the null hypothesis strictly true. Because $\pi_e \neq g$, IS weights are non-trivial, properly stressing the DR components.
* **Metric:** Rejection Rate $= \frac{1}{N_{sim}} \sum \mathbb{I}(|Z| > 1.96)$ (for $\alpha=0.05$).
* **Target:** Rejection rate $\approx 0.05$.

### 4.2. Statistical Power (under $H_1$)
* **Setup:** Gradually increase the policy shift $\Delta \in (0, 1.5]$. $H_1$ is true because $\Delta > 0$ shifts the mean of $\pi_e'$ relative to $\pi_e$, changing the IS weights assigned to the observed $A_1$ values. This reweighting shifts the counterfactual distribution of $A_1$ under $\pi_e'$, which propagates through the backward KRR Q-function estimates to alter the counterfactual distribution of $L_2$ and ultimately $Y$. The observational data $(L_1, A_1, L_2, A_2)$ is unchanged.
* **Metric:** Rejection Rate $= \frac{1}{N_{sim}} \sum \mathbb{I}(|Z| > 1.96)$.
* **Target:** Rejection rate monotonically approaches 1.0 as $\Delta$ increases.

## 5. Concrete Misspecification Grid
We evaluate Type I Error ($\Delta=0$) under four exact nuisance modeling scenarios.

| Scenario | Propensity Models ($\pi_1, \pi_2$) | State-Action Kernel in Backward KRR ($K_H \cdot K_A$) | Expected DR Test Behavior |
| :--- | :--- | :--- | :--- |
| **1. Both Correct** | `GaussianPolicy` parameterized with true linear means ($0.3 \cdot L_t$) | KRR using **RBF Kernel** | Valid (Type I $\approx 0.05$), high efficiency |
| **2. Prop Correct, Out Wrong** | `GaussianPolicy` parameterized with true linear means | KRR using **Linear Kernel** (fails to capture $\sin, \cos$) | Valid (Type I $\approx 0.05$), possibly less efficient |
| **3. Prop Wrong, Out Correct** | `GaussianPolicy` estimating an **intercept only** (omits $L_t$) | KRR using **RBF Kernel** | Valid (Type I $\approx 0.05$), possibly less efficient |
| **4. Both Wrong** | Intercept-only `GaussianPolicy` | KRR using **Linear Kernel** | Invalid (Type I $\gg 0.05$) |

## 6. Codebase Implementation Notes
1. **5D Reshape Fix:** Ensure `LongitudinalOPEData` does not force `Y.reshape(-1, 1)`. The outcome array must retain shape `(n, 5)`. Look specifically inside **both** `_test_cross_fit` and `_test_full` in `nested_dr_kpt.py` to ensure `build_kernel_matrix(data.Y...` handles the multidimensional array without throwing errors.
2. **Logging Policy:** Handled as known (the true DGP behavioral probabilities are passed directly to the estimator to isolate the evaluation of the DR formulation).
3. **`state_kernel` parameter for `NestedDRKPT`:** The state-action kernels inside `_compute_dr_coeff` are currently hardcoded to `"rbf"`. To implement the linear-kernel misspecification in Scenarios 2 and 4, add a `state_kernel` constructor parameter (defaulting to `"rbf"`) and thread it through to the `build_kernel_matrix` calls for `KH` and `KA` in the backward KRR loop.

## 7. Baselines for Comparison
1. **Sequential IPW:** Fails (inflated Type I error) in Scenarios 3 and 4.
2. **Sequential Outcome Regression:** Fails (inflated Type I error) in Scenarios 2 and 4.
3. **Static DR CPME (Single-stage):** Fails to account for treatment-confounder feedback. Its primary failure mode is **loss of statistical power** under $H_1$ ($\Delta > 0$): because the static test ignores the $A_1 \to L_2 \to Y$ causal pathway, it fails to detect the cascading distributional shifts and severely under-rejects compared to the nested test.
