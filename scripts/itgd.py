"""
itgd.py

Author: Eric Scheer
"""

import numpy as np
import scipy.linalg
from covariate_filtering import covariate_filtering

def iterative_threshold_gd_linear(X, y, Sigma, epsilon, eta, w, T=1000, tol=10e-8):
    """
    Implementation of standard gradient descent with iterative hard thresholding on a linear regression
    dataset (X, y) under the strong epsilon contamination model.
    
    Parameters
    ----------
    X : (n, d) float ndarray
        Data matrix with n samples and d features.
    y : (n,) float ndarray
        Label vector.
    Sigma : (d, d) float ndarray
        True covariance matrix of the clean distribution.
    epsilon : float
        Fraction of corrupted samples (0 < epsilon < 0.5).
    eta : float
        Learning rate.
    w : (d,) float ndarray
        Initial weight vector.
    T : int, optional (default=1000)
        Maximum number of gradient descent iterations.
    tol : float, optional (default=1e-8)
        Convergence tolerance. Stops early if gradient norm falls below this value.

    Returns
    -------
    w : (d,) float ndarray
        Final weight vector after convergence or T iterations.
    """

    n = y.shape[0]

    # Data whitening
    # TODO: Add back covariate filtering
    X = X @ scipy.linalg.fractional_matrix_power(Sigma, -0.5)
    #omega = covariate_filtering(X, epsilon)
    omega = np.ones(n)

    for _ in range(T):
        r = np.square(X @ w - y)

        # Hard Thresholding
        k = int(np.ceil(n * (1 - epsilon)))
        thresh_idx = np.argsort(r)[:k]

        X_thresh = X[thresh_idx]
        y_thresh = y[thresh_idx]
        omega_thresh = omega[thresh_idx]

        # Gradient Descent Update
        grad = X_thresh.T @ (omega_thresh * (X_thresh @ w - y_thresh)) / k
        w_new = w - eta * grad

        w = w_new

        # Gradient norm tolerance checking
        if np.linalg.norm(grad) < tol:
            break

    return w

def iterative_threshold_gd_kernel(X, y, Sigma, epsilon, eta, alpha, T=1000, tol=1e-8):
    """
    Implementation of standard gradient descent with iterative thresholding on a linear regression
    dataset (X, y) under the strong epsilon contamination model.
    
    Parameters
    ----------
    X : (n, d) float ndarray
        Data matrix with n samples and d features.
    y : (n,) float ndarray
        Label vector.
    Sigma : (d, d) float ndarray
        True covariance matrix of the clean distribution.
    epsilon : float
        Fraction of corrupted samples (0 < epsilon < 0.5).
    eta : float
        Learning rate.
    alpha : (d,) float ndarray
        Initial alpha iterate.
    T : int, optional (default=1000)
        Maximum number of gradient descent iterations.
    tol : float, optional (default=1e-8)
        Convergence tolerance. Stops early if gradient norm falls below this value.

    Returns
    -------
    w : (d,) float ndarray
        Final weight vector after convergence or T iterations.
    """

    n = y.shape[0]

    # Data whitening
    # TODO: Add back covariate filtering
    X = X @ scipy.linalg.fractional_matrix_power(Sigma, -0.5)
    K = X @ X.T
    #omega = covariate_filtering(X, epsilon)
    omega = np.ones(n)

    for _ in range(T):
        r = np.square(K @ alpha - y)

        # Hard Thresholding
        k = int(np.ceil(n * (1 - epsilon)))
        thresh_idx = np.argsort(r)[:k]

        K_thresh = K[thresh_idx]
        y_thresh = y[thresh_idx]
        omega_thresh = omega[thresh_idx]

        # Gradient Descent Update
        grad = K_thresh.T @ (omega_thresh * (K_thresh @ alpha - y_thresh)) / k
        alpha_new = alpha - eta * grad

        alpha = alpha_new

        # Gradient norm tolerance checking
        if np.linalg.norm(grad) < tol:
            break

    return alpha