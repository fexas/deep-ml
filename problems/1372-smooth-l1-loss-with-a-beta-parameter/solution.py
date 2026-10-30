import numpy as np

def smooth_l1(pred, target, beta=1.0, reduction='mean'):
    """
    Elementwise Smooth L1 loss.

    Args:
        pred, target: arrays of the same shape
        beta: transition point between the quadratic and linear branches (> 0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        float for 'mean'/'sum', array for 'none'
    """
    pred = np.array(pred)
    target = np.array(target)
    abs_diff = np.abs(pred - target)

    if beta == 0:
        raise ValueError("ValueError")

    NextLat = np.where(abs_diff < beta, 0.5 * abs_diff**2 / beta, abs_diff - 0.5 * beta)


    if reduction == 'mean':
        smooth_l1  = np.mean(NextLat)
    elif reduction == 'sum':
        smooth_l1 = np.sum(NextLat)
    elif reduction == 'none':
        smooth_l1 = NextLat
    else:
        raise ValueError("ValueError")
    
    return smooth_l1
