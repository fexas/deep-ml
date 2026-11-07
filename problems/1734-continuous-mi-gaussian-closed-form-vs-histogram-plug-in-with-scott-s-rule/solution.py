import numpy as np

def gaussian_mi(rho):
    
    if np.abs(rho) > 1:
        raise ValueError('ValueError')
    
    mi = np.log(1 / np.sqrt(1-rho**2))

    return float(mi)

def scott_bins(x):
    N = x.shape[0]
    if np.max(x) != np.min(x):
        bins = N**(1/3) * (np.max(x)-np.min(x)) / (3.5 * np.std(x))
        bins = np.ceil(bins)
    else:
        bins = 1
    return int(bins)
def histogram_mi(x, y, bins_x, bins_y, strategy="width"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have the same length")

    if strategy == "width":
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        if xmax == xmin or ymax == ymin:
            return 0.0
        idx_x = np.floor(bins_x * (x - xmin) / (xmax - xmin)).astype(int)
        idx_y = np.floor(bins_y * (y - ymin) / (ymax - ymin)).astype(int)
    elif strategy == "quantile":
        # rank-based equal-frequency binning, using ACTUAL observations
        rx = np.argsort(np.argsort(x))     # rank of each x_i in [0, n)
        ry = np.argsort(np.argsort(y))
        idx_x = np.floor(rx * bins_x / n).astype(int)
        idx_y = np.floor(ry * bins_y / n).astype(int)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    idx_x = np.clip(idx_x, 0, bins_x - 1)
    idx_y = np.clip(idx_y, 0, bins_y - 1)

    # marginals: n-normalized
    px = np.bincount(idx_x, minlength=bins_x) / n
    py = np.bincount(idx_y, minlength=bins_y) / n

    # joint: count PAIRS (idx_x[k], idx_y[k]), n-normalized
    pxy = np.zeros((bins_x, bins_y))
    np.add.at(pxy, (idx_x, idx_y), 1)
    pxy /= n

    def H(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    return H(px) + H(py) - H(pxy.ravel())     
        


    
