import numpy as np

def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""
    # Your code here

    zeros = np.arange(n) * 0.0
    zeros = zeros.reshape(-1,1)
    eye_matirx = zeros @ zeros.T

    index = np.arange(n)
    eye_matirx[index,index] = 1

    return eye_matirx
