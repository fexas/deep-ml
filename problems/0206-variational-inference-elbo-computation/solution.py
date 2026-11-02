import numpy as np

def compute_elbo(x: list[float], q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) for variational inference.
    
    Args:
        x: Observed data points
        q_mean: Mean of variational distribution q(z)
        q_std: Standard deviation of variational distribution q(z)
        prior_mean: Mean of prior distribution p(z)
        prior_std: Standard deviation of prior distribution p(z)
        likelihood_std: Standard deviation of likelihood p(x|z)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        ELBO value (float)
    """
    
    X = np.array(x)
    X = X.reshape(-1,1)

    Z = np.random.normal(loc=q_mean,scale=q_std,size=(1,n_samples))

    # Reconstruction
    recon  = -1 * (X-Z) * (X-Z) / (2 * likelihood_std**2) - 0.5 * np.log(2*np.pi*likelihood_std**2)
    recon =  np.sum(recon, axis = 0)
    recon =  np.mean(recon)

    # E_q log p(z) 
    NEntropy_P = - (Z - prior_mean) * (Z-prior_mean) / (2 * prior_std**2) - 0.5 * np.log(2*np.pi*prior_std**2)
    NEntropy_P = np.mean(NEntropy_P)

    # Entropy H(q)
    Hq = 0.5 * np.log(2 * np.pi * np.e * q_std**2)

    elbo = recon + NEntropy_P + Hq

    return elbo.item()
