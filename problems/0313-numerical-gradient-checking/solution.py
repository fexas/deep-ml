import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    dim = len(x)
    gradient = []
    for i in range(dim):
        x_prank = x.copy()
        x_prank[i] += epsilon
        grad_i  = (f(x_prank)- f(x)) / epsilon
        gradient.append(grad_i)
    numerical_grad = np.array(gradient)

    num_norm = np.linalg.norm(numerical_grad)
    ana_norm = np.linalg.norm(analytical_grad)
    denorm = num_norm + ana_norm

    if denorm == 0:
        relative_error = 0
    else:
        relative_error = np.linalg.norm(numerical_grad-analytical_grad) / denorm

    return numerical_grad, relative_error