
import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    labels, counts = np.unique(y, return_counts=True)
    freq = counts / np.sum(counts)
    val = np.sum(freq * (1-freq))

    return round(val,3)