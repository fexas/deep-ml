def conditional_probability(joint_distribution: dict) -> float:
    """
    Compute conditional probability P(A|B) from a joint probability distribution.

    Args:
        joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

    Returns:
        float: Conditional probability P(A|B)
    """
    evidence_B = 0

    for (a,b), p in joint_distribution.items():
        if b == 'B':
            evidence_B += p

    return joint_distribution[('A','B')] / evidence_B