import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    Y = y.reshape(-1,1)
    W = weights.reshape(-1,1)
    N = len(Y)

    X = X.astype(float)
    Y = Y.astype(float)
    W = W.astype(float)


    if method == 'batch':
        for _ in range(n_epochs):
            grad = - 2 * np.transpose(Y-X @ W) @ X
            W -= learning_rate * grad.reshape(-1,1) / N
    elif method == 'stochastic':
        for _ in range(n_epochs):
            for i in range(len(y)):
                Xi = X[i,:].reshape(-1,1)
                grad = -2 * (Y[i,:] - Xi.T @ W) @ Xi.T
                W -= learning_rate * grad.reshape(-1,1) 
    elif method == 'mini_batch':
        iter_num = len(y) // batch_size - 1
        for _ in range(n_epochs):
            for i in range(0,len(y)-1,batch_size):
                if i < iter_num * batch_size:
                    Xi = X[i:i+batch_size,:]
                    Yi = Y[i:i+batch_size,:]
                    grad = -2 * (Yi - Xi @ W).T @ Xi / batch_size
                    W -= learning_rate * grad.reshape(-1,1) 
                else:
                    Xi = X[i:,:]
                    Yi = Y[i:,:]
                    grad = -2 * (Yi - Xi @ W).T @ Xi / (len(y)-i) 
                    W -= learning_rate * grad.reshape(-1,1) 
    
    return W.flatten()
