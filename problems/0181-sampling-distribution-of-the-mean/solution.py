import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    """
    Compute the mean of sample means to demonstrate the sampling distribution.

    Args:
        num_samples: Number of independent samples to draw
        sample_size: Size of each sample
        distribution: 'uniform' (0,1) or 'exponential' (scale=1)

    Returns:
        Mean of the sample means (float)
    """
    sample_mean_list = []

    for i in range(num_samples):
        if distribution == 'uniform':
            sample = np.random.uniform(0,1, sample_size)
            sample_mean = np.mean(sample).item()
        elif distribution == 'exponential':
            sample = np.random.exponential(1,sample_size)
            sample_mean = np.mean(sample).item()
        sample_mean_list.append(sample_mean)
    
    total_mean = np.mean(np.array(sample_mean_list))

    return total_mean.item()