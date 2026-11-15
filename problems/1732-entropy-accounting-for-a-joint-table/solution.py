import numpy as np

def entropy_table(P, base=2):
    # P: joint table, rows = X, columns = Y
    # return dict with H_X, H_Y, H_XY, H_X_given_Y, H_Y_given_X, I_XY, NMI

    joint_p = P / np.sum(P)
    px = np.sum(joint_p, axis = 1) 
    py = np.sum(joint_p, axis = 0)

    if base > 0:
        nlog_p = np.where(joint_p > 0, -1 * np.log(joint_p) * np.log(np.e) / np.log(base),0)
        nlog_px = np.where(px > 0 , -1 * np.log(px) * np.log(np.e) / np.log(base),0)
        nlog_py = np.where(py > 0, -1 * np.log(py) * np.log(np.e) / np.log(base), 0)
    else: 
        raise ValueError('Wrong Base')


    hx = np.sum(px * nlog_px)
    hy = np.sum(py * nlog_py)
    hxy = np.sum(joint_p * nlog_p) 
    hx_given_y = hxy - hy 
    hy_given_x = hxy - hx
    mi_xy = hx + hy - hxy
    if hx != 0 and hy != 0:
        nmi = mi_xy / np.minimum(hx, hy)
    else:
        nmi = 0.0 
        
    return {
        'H_X': float(hx), 
        'H_Y':float(hy), 
        'H_XY': float(hxy), 
        'H_X_given_Y': float(hx_given_y), 
        'H_Y_given_X': float(hy_given_x), 
        'I_XY': float(mi_xy),
        'NMI': float(nmi)
        }


