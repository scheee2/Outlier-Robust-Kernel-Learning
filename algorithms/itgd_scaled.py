"""
itgd.py — Iterative Thresholding Gradient Descent for robust regression.

Implements:
  - Linear ITGD  (primal space, Algorithm 1 from Rathnashyam & Gittens)
  - Kernel ITGD  (dual space, kernelized extension from Scheer)
  - Huber KRR    (baseline: kernel ridge regression with Huber loss)
"""

import numpy as np
import scipy.linalg as la
from covariate_filtering import covariate_filtering


# ── Activations ──────────────────────────────────────────────────────────────

def _identity(x):
    return x

def _identity_deriv(x):
    return np.ones_like(x)

def _leaky_relu(x, a=0.1):
    return np.where(x >= 0, x, a * x)

def _leaky_relu_deriv(x, a=0.1):
    return np.where(x >= 0, 1.0, a)

_ACTIVATIONS = {
    "identity":   (_identity, _identity_deriv),
    "leaky_relu": (_leaky_relu, _leaky_relu_deriv),
}


# ── Huber loss ──────────────────────────────────────────────────────────────

def _huber_value(r, delta=1.0):
    abs_r = np.abs(r)
    return np.where(abs_r <= delta,
                    0.5 * r ** 2,
                    delta * (abs_r - 0.5 * delta))

def _huber_deriv(r, delta=1.0):
    return np.where(np.abs(r) <= delta, r, delta * np.sign(r))

def get_activation(name):
    """Return (f, f') pair for a named activation."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


# ── Kernel helpers ───────────────────────────────────────────────────────────

def kernel_linear(X1, X2):
    """Linear (dot-product) kernel."""
    return X1 @ X2.T

def kernel_rbf(X1, X2, gamma=None):
    """RBF / Gaussian kernel.  Default gamma = 1/d."""
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    return np.exp(-gamma * np.maximum(sq1 - 2 * X1 @ X2.T + sq2.T, 0))

def kernel_poly(X1, X2, degree=2, coef0=1.0, gamma=None):
    """Polynomial kernel.  Default gamma = 1/d."""
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    return (gamma * X1 @ X2.T + coef0) ** degree


# ── Linear ITGD ─────────────────────────────────────────────────────────────

def itgd_linear(X_train, y_train, Sigma, epsilon, eta=0.1, T=500,
                activation="identity", tol=1e-8):
    """
    Primal-space ITGD with covariate filtering

    Parameters
    ----------
    X_train  : (n, d)  training covariates (possibly corrupted)
    y_train  : (n,)    training labels     (possibly corrupted)
    Sigma    : (d, d)  known covariance matrix of the clean distribution
    epsilon  : float   corruption fraction, 0 < epsilon < 0.5
    eta      : float   learning rate  (paper suggests 0.1 for linear)
    T        : int     max iterations
    activation : str   "identity" | "leaky_relu"
    tol      : float   early-stop on gradient norm

    Returns
    -------
    w_hat : (d,)   learned weight vector in original coordinates
    hist  : dict   keys: grad_norm, mse_full, mse_kept  (lists, one per iter)
    """
    n, d = X_train.shape
    f, fp = get_activation(activation)

    # Whiten covariates and run covariate filtering
    W = la.fractional_matrix_power(Sigma, -0.5).real
    Xw = X_train @ W
    omega = np.asarray(covariate_filtering(Xw, epsilon)).ravel() * n  # scale to ~1

    w = np.zeros(d)
    k = int(np.ceil(n * (1 - epsilon)))
    hist = {"grad_norm": [], "mse_full": [], "mse_kept": []}

    for _ in range(T):
        pred = f(Xw @ w)
        r = (pred - y_train) ** 2
        S = np.argsort(r)[:k]

        z_S = Xw[S] @ w
        resid_S = f(z_S) - y_train[S]
        grad = Xw[S].T @ (omega[S] * resid_S * fp(z_S)) / k

        gn = float(np.linalg.norm(grad))
        hist["grad_norm"].append(gn)
        hist["mse_full"].append(float(np.mean(r)))
        hist["mse_kept"].append(float(np.mean(r[S])))

        w -= eta * grad
        if gn < tol:
            break

    return W @ w, hist


# ── Kernel ITGD ─────────────────────────────────────────────────────────────

def itgd_kernel(X_train, y_train, Sigma, epsilon, eta=0.01,
                kernel="linear", lam=1e-3, T=500,
                activation="identity", tol=1e-8, **kernel_kw):
    """
    Dual-space (kernel) ITGD with covariate filtering

    Parameters
    ----------
    X_train    : (n, d)
    y_train    : (n,)
    Sigma      : (d, d)
    epsilon    : float
    eta        : float   learning rate
    kernel     : str     "linear" | "rbf" | "poly"  — or a callable(X1, X2)
    lam        : float   regularisation strength
    T          : int     max iterations
    activation : str     "identity" | "leaky_relu"
    tol        : float
    **kernel_kw : extra kwargs forwarded to the kernel function
                  (e.g. gamma=0.05 for rbf, degree=3 for poly)

    Returns
    -------
    alpha    : (n,)   dual coefficients
    Xw_train : (n, d) whitened training data  (needed for prediction)
    hist     : dict   keys: grad_norm, mse_full, mse_kept
    """
    # Select kernel function
    if callable(kernel):
        kfn = kernel
    elif kernel == "linear":
        kfn = kernel_linear
    elif kernel == "rbf":
        kfn = lambda X1, X2: kernel_rbf(X1, X2, **kernel_kw)
    elif kernel == "poly":
        kfn = lambda X1, X2: kernel_poly(X1, X2, **kernel_kw)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'")

    n, d = X_train.shape
    f, fp = get_activation(activation)

    # Whiten + filter
    W_half = la.fractional_matrix_power(Sigma, -0.5).real
    Xw = X_train @ W_half
    omega = np.asarray(covariate_filtering(Xw, epsilon)).ravel() * n

    # Build & normalise kernel matrix
    K_raw = kfn(Xw, Xw)
    K_diag = np.sqrt(np.diag(K_raw) + 1e-10)
    K = K_raw / np.outer(K_diag, K_diag) + 1e-6 * np.eye(n)

    alpha = np.zeros(n)
    k = int(np.ceil(n * (1 - epsilon)))
    hist = {"grad_norm": [], "mse_full": [], "mse_kept": []}

    for _ in range(T):
        Ka = K @ alpha
        pred = f(Ka)
        r = (pred - y_train) ** 2
        S = np.argsort(r)[:k]

        z_S = K[S] @ alpha
        resid_S = f(z_S) - y_train[S]
        grad = K[S].T @ (omega[S] * resid_S * fp(z_S)) / k + lam * Ka

        gn = float(np.linalg.norm(grad))
        if gn > 1e6:          # gradient clipping
            grad *= 1e6 / gn
            gn = 1e6

        hist["grad_norm"].append(gn)
        hist["mse_full"].append(float(np.mean(r)))
        hist["mse_kept"].append(float(np.mean(r[S])))

        alpha -= eta * grad
        if gn < tol:
            break

    return alpha, Xw, hist


# ── Huber Kernel Ridge Regression (baseline) ────────────────────────────────

def huber_krr(X_train, y_train, Sigma, kernel="linear", lam=1e-3,
              delta=1.0, eta=0.01, T=500, activation="identity",
              tol=1e-8, **kernel_kw):
    """
    Kernel Ridge Regression with Huber loss — keeps
    RKHS regularisation but replaces the squared loss with Huber

    Unlike ITGD- NO covariate filtering and NO iterative
    thresholding
      So the only robustness mechanism is huber loss

    Good for compariosn ofthe contribution of filtering + thresholding
    when compared against itgd_kernel

    Parameters
    ----------
    X_train, y_train, Sigma  — as in itgd_kernel
    kernel     : "linear" | "rbf" | "poly"  — or a callable(X1, X2)
    lam        : RKHS regularisation strength
    delta      : Huber transition point — smaller -> more robust, more biased
    eta, T, activation, tol, **kernel_kw  — as in itgd_kernel

    Returns
    -------
    alpha    : (n,)   dual coefficients (compatible with predict_kernel)
    Xw_train : (n, d) whitened training data
    hist     : dict   keys: grad_norm, loss, mse_full
    """
    # Select kernel function
    if callable(kernel):
        kfn = kernel
    elif kernel == "linear":
        kfn = kernel_linear
    elif kernel == "rbf":
        kfn = lambda X1, X2: kernel_rbf(X1, X2, **kernel_kw)
    elif kernel == "poly":
        kfn = lambda X1, X2: kernel_poly(X1, X2, **kernel_kw)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'")

    n, d = X_train.shape
    f, fp = get_activation(activation)

    # Whiten (same preprocessing as itgd_kernel for apples-to-apples comparison)
    W_half = la.fractional_matrix_power(Sigma, -0.5).real
    Xw = X_train @ W_half

    # Build & normalise kernel matrix
    K_raw = kfn(Xw, Xw)
    K_diag = np.sqrt(np.diag(K_raw) + 1e-10)
    K = K_raw / np.outer(K_diag, K_diag) + 1e-6 * np.eye(n)

    alpha = np.zeros(n)
    hist = {"grad_norm": [], "loss": [], "mse_full": []}

    for _ in range(T):
        Ka = K @ alpha
        pred = f(Ka)
        resid = pred - y_train

        # Huber gradient:  (1/n) K^T (L_δ'(resid) ⊙ f'(Kα))  +  λ K α
        hd = _huber_deriv(resid, delta)
        grad = K.T @ (hd * fp(Ka)) / n + lam * Ka

        gn = float(np.linalg.norm(grad))
        if gn > 1e6:    # gradient clipping
            grad *= 1e6 / gn
            gn = 1e6

        loss = float(np.mean(_huber_value(resid, delta))
                     + 0.5 * lam * alpha @ Ka)
        hist["grad_norm"].append(gn)
        hist["loss"].append(loss)
        hist["mse_full"].append(float(np.mean(resid ** 2)))

        alpha -= eta * grad
        if gn < tol:
            break

    return alpha, Xw, hist


# ── Prediction ──────────────────────────────────────────────────────────────

def predict_linear(X_test, w, activation="identity"):
    """Predict with a primal-space model."""
    f, _ = get_activation(activation)
    return f(X_test @ w)


def predict_kernel(X_test, Xw_train, alpha, Sigma,
                   kernel="linear", activation="identity", **kernel_kw):
    """
    Predict with a dual-space (kernel) model.

    Uses the same kernel normalisation that was applied during training

    Parameters
    ----------
    X_test   : (m, d)  raw (unwhitened) test covariates
    Xw_train : (n, d)  whitened training covariates  (returned by itgd_kernel)
    alpha    : (n,)    dual coefficients
    Sigma    : (d, d)
    kernel   : str | callable   — must match what was used in training
    activation : str
    """
    if callable(kernel):
        kfn = kernel
    elif kernel == "linear":
        kfn = kernel_linear
    elif kernel == "rbf":
        kfn = lambda X1, X2: kernel_rbf(X1, X2, **kernel_kw)
    elif kernel == "poly":
        kfn = lambda X1, X2: kernel_poly(X1, X2, **kernel_kw)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'")

    W_half = la.fractional_matrix_power(Sigma, -0.5).real
    Xw_test = X_test @ W_half

    K_raw_te_tr = kfn(Xw_test, Xw_train)
    K_diag_tr   = np.sqrt(np.diag(kfn(Xw_train, Xw_train)) + 1e-10)
    K_diag_te   = np.sqrt(np.diag(kfn(Xw_test, Xw_test))   + 1e-10)
    K_norm      = K_raw_te_tr / np.outer(K_diag_te, K_diag_tr)

    f, _ = get_activation(activation)
    return f(K_norm @ alpha)
