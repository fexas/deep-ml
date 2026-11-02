import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    
    if distribution == 'uniform':
        samples = np.random.uniform(0,1,size=(runs,n))
        population_mean = 0.5
        population_std =  np.sqrt(1 / 12) 
    elif distribution == 'exponential':
        samples = np.random.exponential(1.0,size=(runs,n))
        population_mean = 1.0
        population_std = 1.0
    elif distribution == 'bernoulli':
        samples = (np.random.rand(runs,n)<0.3).astype(float)
        population_mean = 0.3
        population_std = np.sqrt(0.3 * 0.7)
    else:
        raise ValueError('ValueError')
    
    samples_mean = np.mean(samples,axis=1)

    z_score_samples = (samples_mean - population_mean) / (population_std / np.sqrt(n))

    z_mean = np.mean(z_score_samples).item()
    z_std  = np.std(z_score_samples).item()

    return {
        'mean': z_mean,
        'std':  z_std

    }