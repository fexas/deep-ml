import numpy as np

def mini_batch_gd_step(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    batch_indices: list,
    lr: float,
) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """

    constant = np.ones((X.shape[0], 1))
    X_expand = np.concatenate((X, constant), axis=1)
    beta = np.append(weights, bias)
    beta = beta.reshape(-1, 1)

    batch_indices = np.asarray(batch_indices)
    X_batch = X_expand[batch_indices]
    y_batch = y[batch_indices].reshape(-1, 1)

    m = X_batch.shape[0]

    grad = -2 * X_batch.T @ (y_batch - X_batch @ beta) / m

    beta -= lr * grad

    return beta.flatten()  
    

