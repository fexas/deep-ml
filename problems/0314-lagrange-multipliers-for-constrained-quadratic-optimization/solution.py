import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    
    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    
    Args:
        Q: 2x2 symmetric positive definite matrix
        c: 2-element vector (linear coefficients)
        a: 2-element vector (constraint coefficients)
        b: scalar (constraint value)
    
    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """

    a = a.reshape(-1,1)
    c = c.reshape(-1,1)
    b = np.array([[b]])
    
    if np.linalg.norm(a, ord=2) == 0:
        raise ValueError('ValueError')
    
    Q_inv = np.linalg.inv(Q)
    Lambda_nu = b + a.T @ Q_inv @ c
    Lambda_de = a.T @ Q_inv @ a 
    Lambda = Lambda_nu / Lambda_de

    X_sol = np.linalg.solve(Q, -1 * c + Lambda[0,0] * a)
    Object_value =  X_sol.T @ Q @ X_sol / 2 + c.T @ X_sol
    Object_value = np.round(Object_value.ravel(),4).tolist()
    Object_value = Object_value[0]
    X_sol =  np.round(X_sol.ravel(),4).tolist()

    return{
        'x': X_sol,
        'lambda': round(Lambda[0,0],4),
        'objective': Object_value

    }


