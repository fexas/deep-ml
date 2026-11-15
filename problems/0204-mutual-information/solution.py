import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information between two random variables.
    
    Args:
        joint_prob: 2D joint probability distribution P(X,Y)
    
    Returns:
        Mutual information I(X;Y)
    """
    
    joint_prob = np.array(joint_prob)
    marginal_px = np.sum(joint_prob, axis=1)
    marginal_py = np.sum(joint_prob, axis=0)

    nlog_joint_prob = np.where(joint_prob > 0, -1 * np.log(joint_prob), 0)
    nlog_px = np.where(marginal_px >0 , -1 * np.log(marginal_px), 0)
    nlog_py = np.where(marginal_py > 0 , -1 * np.log(marginal_py), 0)

    hx = np.sum(marginal_px * nlog_px)
    hy = np.sum(marginal_py * nlog_py)
    hxy = np.sum(joint_prob * nlog_joint_prob)

    mi = hx + hy - hxy

    return mi
