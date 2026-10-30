import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    Y = y.reshape(-1,1)
    W = w0.reshape(-1,1)
    N = len(y)
    if n_steps >1 :
        lr_list = [10**(a + (b-a)*k/(n_steps-1)) for k in range(n_steps)]
    else:
        lr_list = [10**b]

    MSE_list = []

    for lr in lr_list:
        grad = -2 * (Y- X @ W).T @ X / N
        W -= lr * grad.reshape(-1,1)
        MSE_kth = (Y- X @ W).T @ (Y-X @ W) / N
        MSE_list.append(MSE_kth.item())
    
    return MSE_list
    


