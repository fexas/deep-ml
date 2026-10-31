import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    
    val = np.linalg.eigvals(hessian)

    diag_pd = np.where(val > tol, 1.0, 0.0)
    diag_nd = np.where(val < -1 * tol, 1.0, 0.0)
    diag_all_zero = np.where(np.abs(val) < tol, 1.0, 0.0)

    if np.sum(diag_all_zero, axis=0) >= 1:
        return None
    elif np.prod(diag_pd) == 1:
        diag = -1
    elif np.prod(diag_nd) == 1:
        diag = 1
    else:
        diag = 0

    return diag