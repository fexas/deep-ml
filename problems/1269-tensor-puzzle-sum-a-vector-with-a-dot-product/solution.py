import numpy as np

def vector_sum(a: np.ndarray):
    """Sum elements of 1-D array a without np.sum / loops."""
    # Your code here
    
    dim = a.shape[0]
    ones = np.arange(dim) * 0.0 + 1.0

    return float( a @ ones)

