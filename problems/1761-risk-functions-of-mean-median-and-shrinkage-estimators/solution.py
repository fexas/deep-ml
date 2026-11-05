import numpy as np
from math import pi

def risk_closed_form(est, theta_star, N, sigma=1.0):
    if est['name'] == 'mean':
        risk = sigma**2 / N
    elif est['name'] == 'median':
        risk = (pi * sigma**2) / (2 * N)
    elif est['name'] == 'const':
        risk = (est['theta0']-theta_star)**2
    elif est['name'] == 'postmean':
        kappa = est['kappa']
        weight = N / (N + kappa)
        bias_sq = ( (1-weight) *(est['theta0']-theta_star) ) **2
        variance = (N * sigma**2) / (N + kappa)**2
        risk = bias_sq + variance
    else:
        raise ValueError('ValueError')

    return float(risk)

def risk_mc(est, theta_star, N, n_rep, seed, sigma=1.0):
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n_rep, N)) * sigma + theta_star
    
    if est['name'] == 'mean':
        mean = np.mean(samples, axis=1)
        risk = np.mean( (mean-theta_star)**2)
    elif est['name'] == 'median':
        median = np.median(samples,axis=1)
        risk = np.mean( (median-theta_star)**2 )
    elif est['name'] == 'const':
        risk = (est['theta0'] - theta_star)**2
    elif est['name'] == 'postmean':
        kappa = est['kappa']
        weight = N / (N + kappa)
        mean = np.mean(samples, axis=1)
        postmean = weight * mean + (1-weight)*est['theta0']
        risk = np.mean((postmean - theta_star)**2)
    else: 
        raise ValueError('ValueError')
    
    return float(risk)


def bayes_risk(est, N, tau, sigma=1.0, prior_mean=0.0):
    if est['name'] == 'mean':
        bayes_risk = sigma**2 / N
    elif est['name'] == 'median':
        bayes_risk = (pi * sigma**2) / (2 * N)
    elif est['name'] == 'const':
        bayes_risk = tau**2 + (est['theta0']-prior_mean)**2
    elif est['name'] == 'postmean':
        kappa = est['kappa']
        theta0 = est['theta0']
        weight = N / (N + kappa)
        bayes_risk = (1-weight)**2 * ( (theta0-prior_mean)**2  + tau**2) + (N * sigma**2 ) / (N + kappa)**2
    else:
        raise ValueError('ValueError')
    
    return float(bayes_risk)

def minimax_index(ests, N, theta_grid, sigma=1.0):
    
     
    for i, est in enumerate(ests):

        for j, th in enumerate(theta_grid):
            risk = risk_closed_form(est, th, N, sigma)
            if j == 0:
                max_risk = risk
            else:
                if risk > max_risk:
                    max_risk = risk
        
        if i == 0:
            minmax_index = i
            minmax_risk = max_risk
        else:
            if max_risk < minmax_risk:
                minmax_index = i 
                minmax_risk = max_risk

    return minmax_index

        


    
