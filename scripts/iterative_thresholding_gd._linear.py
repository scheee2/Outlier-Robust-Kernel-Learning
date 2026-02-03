"""
iterative_thresholding_gd_linear.py

Author: Eric Scheer
"""

import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt

def weighted_mean(X, w):
    """
    Calculates a weighted mean of the data in X
    
    :param X: n x d data matrix in row form
    :param w: weight vector of length d where 0 <= w_i <= 1 for each weight

    :returns weighted_mean:
    """

    return (X.T @ w) / np.sum(w)

def weighted_covariance(X, w):
    """
    Calculates a weighted covariance matrix for the data in X
    
    :param X: Description
    :param w: Description
    """

    mu = weighted_mean(X, w)
    X_centered = X - mu

    return ((X_centered.T * w) @ X_centered) / np.sum(w)

def mmw_update(d, feedback, alpha):
    """
    Applys the MMW update rule described in https://arxiv.org/pdf/1906.11366
    
    :param d: number of data features
    :param feedback: List of d x d feedback matrices
    :param alpha: mmw parameter
    """

    # Scaled sum of feedback matrices
    F_sum = alpha * np.sum(feedback, axis=0)
    if len(feedback) == 0:
        F_sum = alpha * np.eye(d)

    U = scipy.linalg.expm(F_sum)
    return U / np.linalg.trace(U)

def filter_1d(w, tau, b):
    """
    Implementation of algorithm 8 (1DFilter) described in https://arxiv.org/pdf/1906.11366

    :param w: Weight vector of length m
    :param tau: Nonnegative vector QUE scores tau_1... tau_m
    :param b: Multiplicative threshold parameter.
    """
    
    tau_max = np.max(tau)
    sigma = np.sum(w * tau)
    t_max = max(int(tau_max // (np.exp(b * sigma))), 1)

    # Create vector of F_t values
    geometric_scalars = (1 - (tau / tau_max))

    F = np.zeros(t_max)
    w_t = w.copy()
    for t in range(t_max):
        F[t] = np.sum(w_t * tau)
        w_t *= geometric_scalars

    # Find smallest t such that F_t <= b * sigma
    t = np.searchsorted(-F, -b * sigma)
    t = min(t, t_max - 1) # index clamping

    return geometric_scalars**t * w

def covariate_filtering(X, epsilon, C=5, max_epochs=100):
    """
    Implementation of algorithm 4 described in https://arxiv.org/pdf/1906.11366
    
    :param X: n x d data matrix with (1 - epsilon) * n points sampled from a subgaussian distribution.
    :param epsilon: Description
    :param C: Sufficiently large universal constant
    """

    n, d = X.shape
    w = np.ones(n) / n

    for s in range(max_epochs):
        cov = weighted_covariance(X, w)
        lmbda = np.linalg.norm(cov - np.eye(d), ord=2)

        # Breaking Condition
        if lmbda < C * epsilon:
            return w
        
        alpha = 1 / (1.1 * lmbda)

        # Gain matrices used for MMW updates
        feedback = []

        for t in range(int(np.log(d)) + 2):
            cov_t = weighted_covariance(X, w)
            lmbda_t = np.linalg.norm(cov_t - np.eye(d), ord=2)

            # Terminate Epoch
            if lmbda_t <= 0.5 * lmbda:
                break

            # Perform MMW Update
            U_t = mmw_update(d, feedback, alpha)

            # Compute score oracles tau_t and q_t
            # tau values do not need to be computed unless q_t > lmbda / 5
            # TODO: implement linear approximate score computations from Algorithm 5 of https://arxiv.org/pdf/1906.11366

            # q_t =〈cov_t - eye(d), U_t〉
            q_t = np.linalg.trace((cov_t - np.eye(d)).T @ U_t)

            if q_t > lmbda / 5:
                # For i = 0 ... n, tau_t,i = (x_i - mu_t)^T U_t (x_i - mu_t)
                # Use einsum to calculate quadratic forms
                mu_t = weighted_mean(X, w)
                X_centered_t = X - mu_t
                tau_t = np.einsum('ij,ik,jk->i', X_centered_t, X_centered_t, U_t) 

                # sort tau scores in descending order and pick the first m sorted weights such that
                # sum(w_i + ... + w_m) >= 2 * epsilon
                desc_idx = np.argsort(tau_t)[::-1]
                m = np.searchsorted(np.cumsum(w[desc_idx]), 2 * epsilon, side='right')

                update_idx = desc_idx[:m]
                w[update_idx] = filter_1d(w[update_idx], tau_t[update_idx], 0.25)

            feedback.append(weighted_covariance(X, w))

    return w

def iterative_threshold_gd_linear(X, y, Sigma, epsilon, eta, w, T=1000):
    """
    Implementation of standard gradient descent with iterative thresholding on a linear regression
    dataset (X, y) under the strong epsilon contamination model. 
    
    :param X: n x d data matrix given in row form.
    :param y: Label vector.
    :param Sigma: True covariance matrix of clean data.
    :param epsilon: Fraction of data that is corrupted.
    :param eta: Gradient descent learning rate.
    :param w: Initial weights.
    :param T: Number of iteration. 1000 by defaukt.
    """

    n = y.shape[0]
    weight_iterates = [w.copy()]

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
        grad = (1 / k) * omega_thresh * X_thresh.T @ (X_thresh @ w - y_thresh)
        w_new = w - eta * grad

        weight_iterates.append(w_new.copy())

        w = w_new

    return w, np.array(weight_iterates)

def iterative_threshold_gd_kernel(X, y, Sigma, epsilon, eta, alpha, lam, T=1000):
    """
    Implementation of standard gradient descent with iterative thresholding on a linear regression
    dataset (X, y) under the strong epsilon contamination model. 
    
    :param X: n x d data matrix given in row form.
    :param y: Label vector.
    :param Sigma: True covariance matrix of clean data.
    :param epsilon: Fraction of data that is corrupted.
    :param eta: Gradient descent learning rate.
    :param alpha: Initial alpha.
    :param lam: Regularization parameter.
    :param T: Number of iteration. 1000 by defaukt.
    """

    n = y.shape[0]
    alpha_iterates = [alpha.copy()]

    # Data whitening
    # TODO: Add back covariate filtering
    X = X @ scipy.linalg.fractional_matrix_power(Sigma, -0.5)

    K = X @ X.T

    for _ in range(T):
        r = np.square(K @ alpha - y)

        # Hard Thresholding
        k = int(np.ceil(n * (1 - epsilon)))
        pad_idx = np.argsort(r)[k:]

        # Pad out K and labels for data
        K[pad_idx, :] = 0
        K[:, pad_idx] = 0

        # Gradient Descent Update
        grad = (1 / k) * K @ (K @ alpha - y) + lam * K @ alpha
        alpha_new = alpha - eta * grad

        alpha_iterates.append(alpha_new.copy())

        alpha = alpha_new

    return alpha, np.array(alpha_iterates)

def main():
    # Generate clean linear regression dataset of size (n_train)
    # Clean model given by y = w_*^T x + xi where xi ~ N(0, sigma^2)
    np.random.seed(1)
    n_train = 500
    d = 50
    epsilon = 0.49
    label_noise_std = 0.1
    label_corruption_std = 5
    Sigma = np.eye(d)

    X_clean = np.random.normal(0, 1, size=(n_train, d))
    w_true = np.random.normal(0, 1, d)
    alpha_true = np.linalg.pinv(X_clean.T) @ w_true

    y_clean = X_clean @ w_true + np.random.normal(0, label_noise_std, n_train)


    # Corrupt n * epsilon samples (covariates and labels)
    # For now just adding large gaussian noise
    # TODO: add covariate corrupting
    num_outliers = int(epsilon * n_train)
    outlier_idx = np.random.choice(n_train, num_outliers, replace=False)

    X_corrupted = X_clean.copy()
    # X_corrupted[outlier_idx] += 3 * np.random.normal(0, 1, size=(num_outliers, d))

    y_corrupted = y_clean.copy()
    y_corrupted[outlier_idx] += np.random.normal(0, label_corruption_std, num_outliers)

    # Hyperparameters
    eta = 0.1
    w_init = np.zeros(d)
    alpha_init = np.zeros(n_train)

    # Learned weights and alphas
    w_hat, w_iterates = iterative_threshold_gd_linear(X_corrupted, y_corrupted, Sigma, epsilon, eta, w_init)
    alpha_hat, alpha_iterates = iterative_threshold_gd_kernel(X_corrupted, y_corrupted, Sigma, epsilon, 0.001, alpha_init, lam=0.01)
    
    # Generate clean test data
    n_test = 200
    X_test = np.random.randn(n_test, d)
    y_test = X_test @ w_true + np.random.randn(n_test)

    w_parameter_errors = np.linalg.norm(w_iterates - w_true, axis=1)
    w_test_errors = np.linalg.norm(X_test @ w_iterates.T - y_test[:, None], axis=0)
    alpha_parameter_errors = np.linalg.norm(alpha_iterates - alpha_true, axis=1)
    alpha_test_errors = np.linalg.norm(X_test @ X_corrupted.T @ alpha_iterates.T - y_test[:, None], axis=0)

    plt.figure(figsize=(10, 4))

    # Test Error Plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(w_test_errors, color='r', label=r'Linear Regression')
    ax1.plot(alpha_test_errors, color='b', label=r'Linear KRR')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Test Error")
    ax1.set_title("Test Error per Iteration")
    ax1.legend()
    ax1.grid()

    ax1.spines['left'].set_linewidth(2)
    ax1.axhline(y=0, color='black', linewidth=2) 
    ax1.set_ylim(bottom=-0.1)

    # Parameter Error Plot
    ax2 = plt.subplot(1, 2, 2) 
    ax2.plot(w_parameter_errors, color='r', label=r'Linear Regression')
    ax2.plot(alpha_parameter_errors, color='b', label=r'Linear KRR')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Parameter Error")
    ax2.set_title("Linear Regression Parameter Error per Iteration")
    ax2.grid()
    ax2.legend()

    ax2.spines['left'].set_linewidth(2)
    ax2.axhline(y=0, color='black', linewidth=2) 
    ax2.set_ylim(bottom=-0.01)
    
    plt.suptitle(f"IGD Errors with $\\epsilon = {epsilon}$")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4) 
    plt.show()

if __name__ == '__main__':
    main()