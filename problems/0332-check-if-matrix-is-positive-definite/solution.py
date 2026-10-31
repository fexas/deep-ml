import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # for 2 by 2 matirx
    # a = matrix[0][0]
    # b = matrix[0][1]
    # c = matrix[1][0]
    # d = matrix[1][1]

    # if (a+d) > 2e-10 and (a*d - b*c) > 1e-20:
    #     is_positive_definite = True
    # else:
    #     is_positive_definite = False
    
    # eigvals_list = [0] * 2

    # delta = (a+d)**2/4 - (a*d-b*c)

    # if delta >= 0:
    #     eigvals_list[0] = (a+d)/2 - np.sqrt(delta)
    #     eigvals_list[1] = (a+d)/2 + np.sqrt(delta)
    # else:
    #     raise ValueError('ValueError')

    # for 2 dimensional matrix

    eigvals_list = np.linalg.eigvals(matrix)
    eigvals_list = np.real_if_close(eigvals_list, tol=1e-10)
    pd_diag = np.where(eigvals_list > 1e-10, 1.0, 0.0)

    if np.iscomplexobj(eigvals_list):
        raise ValueError('ValueError')
    
    if np.prod(pd_diag) > 0:
        is_positive_definite = True
    else:
        is_positive_definite = False
    
    eigvals_list = np.sort(eigvals_list, axis=0).tolist()

       
    return{
        'is_positive_definite': is_positive_definite,
        'eigenvalues': [round(eigen,4) for eigen in eigvals_list]
    }
