import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here

    n = X.shape[0]
    d =  X.shape[1]
    w = np.zeros((d,1))
    b = 0.0

    for i in range(max_iter):
        res = X @ w + b - y.reshape(-1,1)
        res = res.reshape(-1,1)
        grad_w = X.T @ res / n + alpha1 * np.sign(w) + 2 * alpha2 * w
        grad_b = np.mean(res).item()
        
        if np.linalg.norm(grad_w, ord=1) < tol:
            break
        else:
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
        
    return w.flatten().tolist() , b