import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    mu_p = mu_p.reshape(-1,1)
    mu_q = mu_q.reshape(-1,1)
    d = float(mu_p.shape[0])
    mu_diff = mu_p-mu_q

    KL = 0.5 * (-1 * d + np.trace(np.linalg.inv(Cov_q) @ Cov_p) + mu_diff.T @ np.linalg.inv(Cov_q) @ mu_diff - np.log((np.linalg.det(Cov_p) / np.linalg.det(Cov_q))) )

    return KL.item()