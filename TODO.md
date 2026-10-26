# TODO

Open work on `experiments/robust_rff_itgd.ipynb`. Background and current settings are in `CLAUDE.md`.

## 1. Increase the number of RFFs until clean RFF matches clean KRR

**Problem.** At the cross-validated λ, `oracle clean RFF` was about 1.4× worse than `clean KRR` at ε = 0 with
m = ⌈√n log n⌉ = 340 (measured before §1 switched to the Hermite construction; re-measure). The √n log n rule (Rudi & Rosasco 2017) assumes λ ≈ 1/√n; the CV λ is 30–70× smaller, which
needs more features. A quick test gave a gap of 1.19× at m = 1000 and 1.05× at m = 3000.

**Approach.**
- Scale m with the chosen λ instead of n: m = C · N(λ) log N(λ), where N(λ) = Σ_α μ_α/(μ_α + λ) is now available in
  closed form from `mu_modes` (§1). Pick C from a sweep.
- Acceptance: `oracle clean RFF` / `clean KRR` ≤ 1.05 at ε = 0, averaged over ≥ 5 trials.
- Cost to fix first: `itgd_linear` calls `fractional_matrix_power` on an m × m matrix three times per trial. Replace it
  with one `eigh` of M̂ + λ_w I per trial; at m ≈ 3000 the current code is slow.

## 2. Bound the labels

**Problem.** The noise is Gaussian, so y is unbounded. B_y = sup|f*| + 3s is only a nominal bound used by the
adversaries, and neither ITGD nor the estimators enforce any bound on y. Under `gross`, y = 100 B_y reaches the fit.

**Options.**
- Exact bound on f*: for the Gaussian kernel, ‖f‖_∞ ≤ ‖f‖_H, and ‖f*‖²_H = Σ_α ⟨f*, φ_α⟩²/μ_α is computable from
  `f_coef` and `mu_modes`. With sub-Gaussian noise, B_y = ‖f*‖_H + s√(2 log(2n/δ)) holds for all clean labels with
  probability ≥ 1 − δ.
- Data-driven bound from half A only (keeps the split clean): clip y to median(y_A) ± C · MAD(y_A).
- Change the noise model to a bounded one (e.g. truncated Gaussian) so B_y holds deterministically, if the theory needs
  bounded labels.
- Apply the chosen bound by clipping y in half B before ITGD, and report how much of each attack it neutralizes on its
  own.

## 3. Add a Friedman / real-world benchmark

**Friedman #1** (Friedman 1991; `sklearn.datasets.make_friedman1`): y = 10 sin(πx₁x₂) + 20(x₃ − 0.5)² + 10x₄ + 5x₅ + noise,
x ~ U[0, 1]^d with d ≥ 5. f* is known, so L² errors can still be computed on an evaluation sample. There is no closed-form T
or source condition, and X is uniform, so choose σ by the median heuristic or CV.

**Real data.** `data/social_media_productivity_6000.csv` is already in the repo: 6000 rows, 7 numeric features plus
`addiction_level` (categorical), target `productivity_score`. It is censored (target has zeros, `focus_score` is
capped at 100), so check the fit on clean data first. Standardize the features. With no f*, report held-out test MSE on
clean data, and inject corruption with `corrupt_inputs` on the training split only.

**Refactor needed.** `run_trial` reads `sample_xy`, `X_eval` and `f_eval` as globals. Pass a dataset object in
instead, so the same trial code runs on the synthetic, Friedman and real-data settings.
