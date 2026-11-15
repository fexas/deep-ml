import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement

    def gini_split(feature, y):
        # return gini index, threshold
        index = np.argsort(feature)
        fea_sort = feature[index]
        y_sort = y[index]

        best_gini_index = 0
        best_threshold = 0.0
        n = feature.shape[0]
        for i in range(n):
            if i == 0:
                labels, counts = np.unique(y_sort[i+1:], return_counts=True)
                freqs = counts / np.sum(counts)
                best_threshold = fea_sort[0]
                best_gini_index = np.sum(freqs * (1-freqs)) * (n-1) / n
            elif i < n - 1:
                fromer_labels, former_counts = np.unique(y_sort[:i+1], return_counts=True)
                latter_labels, latter_counts = np.unique(y_sort[i+1:], return_counts=True)
                former_freqs = former_counts / np.sum(former_counts)
                latter_freqs = latter_counts / np.sum(latter_counts)
                n
                former_weights = i + 1 / n
                latter_weights = (n-i-1) / n
                gini_index = former_weights * np.sum(former_freqs * (1-former_freqs)) + latter_weights * np.sum(latter_freqs * (1-latter_freqs))

                if gini_index < best_gini_index:
                    best_threshold = fea_sort[i]
                    best_gini_index = gini_index
            else:
                labels, counts = np.unique(y_sort, return_counts=True)
                freqs = counts / np.sum(counts)
                gini_index = np.sum(freqs * (1-freqs))
                if gini_index < best_gini_index:
                    best_threshold = fea_sort[i]
                    best_gini_index = gini_index

        return best_threshold, best_gini_index

    d = X.shape[1]
    best_feature_index = 0
    best_index = 0
    best_thre = 0
    best_gin = 0
    for i in range(d):
        if i == 0:
            best_index = i
            best_thre, best_gin = gini_split(X[:,i],y)
        else:
            thre, gin = gini_split(X[:,i],y)
            if gin < best_gin or best_gin == 0:
                best_index = i
                best_thre = thre
                best_gin = gin
    
    return best_index, best_thre
