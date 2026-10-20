import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """

    n = y.shape[0]
 

    for i in range(n_iter):

        index = i % n
        xi = X[index]
        xi = xi.reshape(-1,1)
        weights = weights.reshape(-1,1)

        grad = -2 *  xi * (y[index] - xi.T @ weights) 

        weights -= learning_rate * grad 

    
    return weights.flatten()


