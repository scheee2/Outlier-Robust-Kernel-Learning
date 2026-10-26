# Outlier-Robust-Kernel-Learning

Robust kernel regression when an ε-fraction of training pairs may be adversarially corrupted.

## Layout

```
algorithms/                  Python package with the methods
  covariate_filtering.py     Covariate filtering (Dong et al. 2019)
  itgd.py                    Linear/kernel ITGD and Huber KRR baseline
  itgd_scaled.py             ITGD variant used by itgd_experiments.ipynb
                             (scales ω by n and the kernel gradient by 1/k)
experiments/                 Jupyter notebooks
  itgd_experiments.ipynb     ITGD vs. Huber KRR / sklearn KRR under corruption
  covariate_filtering_test.ipynb
  random_features_test.ipynb Random Fourier features experiments
data/                        Datasets
docs/                        Report, slides, and project prospectus (LaTeX)
```

## Setup

```
pip install -r requirements.txt
```

Notebooks add the repo root to `sys.path` so they can `import algorithms` when run from `experiments/`.
