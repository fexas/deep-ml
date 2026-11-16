import numpy as np

def entropy_split_selection(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """

    n_samples, n_features = X.shape

    def entropy_feature_split(feature, y):
        fea_labels, fea_counts = np.unique(feature, return_counts=True)
        if fea_labels.shape[0] == 1:
            best_fea_thre = 0.0
            best_fea_info_gain = 0.0
        else:
            index = np.argsort(feature)
            fea_sort = feature[index]
            y_sort = y[index]

            def cal_entropy(y):
                yy_labels, yy_counts = np.unique(y,return_counts=True)
                yy_freqs = yy_counts / np.sum(yy_counts)
                nlog_yy_freqs = np.where(yy_freqs > 0, -1 * np.log2(yy_freqs), 0)
                ent = np.sum(yy_freqs * nlog_yy_freqs)
                return ent

            init_ent = cal_entropy(y_sort)

            best_fea_thre = 0.0
            best_fea_info_gain = 0.0

            for i in range(n_samples-1):
                fea_thre = (fea_sort[i] + fea_sort[i+1])/ 2
                left_y = y_sort[:i+1]
                right_y = y_sort[i+1:]
                left_ent = cal_entropy(left_y)
                right_ent = cal_entropy(right_y)
                weight_l = (i+1) / n_samples
                weight_r = 1 - weight_l
                current_ent = weight_l * left_ent + weight_r * right_ent
                current_info_gain = round(init_ent - current_ent,4)
                if i == 0:
                    best_fea_info_gain = current_info_gain
                    best_fea_thre = fea_thre
                elif current_info_gain > best_fea_info_gain:
                    best_fea_info_gain = current_info_gain
                    best_fea_thre = fea_thre
                else:
                    continue
        
        return best_fea_thre, best_fea_info_gain

    best_feature_index = 0
    best_threshold = 0.0
    best_info_gain = 0.0

    for i in range(n_features):
        thre, info_gain = entropy_feature_split(X[:,i],y)
        info_gain = round(info_gain,4)
        if i == 0:
            best_threshold = thre
            best_info_gain = info_gain
        elif info_gain > best_info_gain:
            best_feature_index = i
            best_threshold = thre
            best_info_gain = info_gain
    
    return best_feature_index, round(best_threshold,4), round(best_info_gain,4)
        

                


    