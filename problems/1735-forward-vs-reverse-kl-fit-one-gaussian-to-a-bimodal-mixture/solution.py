import numpy as np

def fit_forward_kl(grid, p):
    # return (mu, sigma, kl_forward)

    p_partition = np.sum(p)
    p = p / p_partition
    
    mu = np.sum(grid * p)
    normalized_grid = grid - mu
    sigma = np.sqrt(np.sum( normalized_grid * normalized_grid * p))
    log_p = np.where(p > 0, np.log(p), 0)

    q = np.exp(-1 * normalized_grid * normalized_grid /(2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    q = q / np.sum(q)
    log_q = np.log(q)

    kl_forward = np.sum(p * (log_p - log_q))

    return float(mu), float(sigma), float(kl_forward)

def fit_reverse_kl(grid, p, mus, sigmas):
    # return (mu, sigma, kl_reverse) of the best candidate

    p_partition = np.sum(p)
    p = p / p_partition

    minimum_kl_reverse = np.inf
    
    for i, mu in enumerate(mus):
        for j, sigma in enumerate(sigmas):
            normalized_grid = grid - mu
            q = np.exp(-1 * normalized_grid * normalized_grid /(2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
            q = q / np.sum(q)
            log_q = np.where(q>0, np.log(q),0)
            log_q_minus_log_p = np.where(p > 0, log_q - np.log(p), np.inf)

            kl_reverse = np.sum( q * log_q_minus_log_p)

            if i == 0 and j == 0:
                minimum_kl_reverse = kl_reverse
                best_mu = mu
                best_sigma = sigma
            else:
                if kl_reverse < (minimum_kl_reverse - 1e-9):
                    minimum_kl_reverse = kl_reverse
                    best_mu = mu
                    best_sigma = sigma
    
    return float(best_mu), float(best_sigma), float(minimum_kl_reverse)




