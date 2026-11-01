import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    
    k = np.sum(np.array(observations))
    n = len(observations)
    nu = alpha + k - 1
    de = alpha + beta + n - 2
    alpha2 = alpha + k
    beta2 = n - k + beta

    
    if alpha2 > 1 and beta2 > 1:
        theta_map = round(nu / de,4) 
    elif alpha2 == 1 and beta2 > 1:
        theta_map = round(0.0,4)
    elif alpha2 > 1 and beta2 == 1:
        theta_map = round(1.0,4)
    elif alpha2 == 1 and beta2 == 1:
        theta_map = '[0,1]'
    else:
        raise ValueError('ValueError')
    
    return theta_map
    
