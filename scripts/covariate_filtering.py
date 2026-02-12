"""
covariate_filtering.py

Implements the Covariate Filtering Algorithm described in https://arxiv.org/pdf/1906.11366

Author: Eric Scheer
"""

import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt

def weighted_mean(X, w):
    """
    Calculates a weighted mean of the data in X
    
    Parameters
    ----------
    X : n x d data matrix in row form
    w : weight vector of length d where 0 <= w_i <= 1 for each weight

    Returns
    -------
    weighted_mean:
    """

    return (X.T @ w) / np.sum(w)

def weighted_covariance(X, w):
    """
    Calculates a weighted covariance matrix for the data in X
    
    Parameters
    ----------
    X : Description
    w : Description
    """

    mu = weighted_mean(X, w)
    X_centered = X - mu

    return ((X_centered.T * w) @ X_centered) / np.sum(w)

def mmw_update(d, feedback, alpha):
    """
    Applys the MMW update rule described in https://arxiv.org/pdf/1906.11366
    
    Parameters
    ----------
    d : number of data features
    feedback : List of d x d feedback matrices
    alpha : mmw parameter
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

    Parameters
    ----------
    w : Weight vector of length m
    tau : Nonnegative vector QUE scores tau_1... tau_m
    b : Multiplicative threshold parameter.
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
    
    Parameters
    ----------
    X : n x d data matrix with (1 - epsilon) * n points sampled from a subgaussian distribution.
    epsilon : Description
    C : Sufficiently large universal constant
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