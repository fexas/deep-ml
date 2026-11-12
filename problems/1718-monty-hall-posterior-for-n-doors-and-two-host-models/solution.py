import numpy as np

def monty_posterior(n_doors, chosen, opened, host):
    # host in {"knows", "random"}; doors numbered from 0
    # return list of n_doors posterior probabilities

    prior = np.ones(n_doors) / n_doors

    def multi_prod(s):

        prod = 1 
        if s > 0:
            for i in range(s):
                prod *= (i+1)
        elif s == 0:
            prod = 1
        else:
            raise ValueError('ValueError')
        
        return prod

    def combination_number(m,k):
        return multi_prod(m) / (multi_prod(k) * multi_prod(m-k))

    if host == "knows":

        likelihood_given_hi = np.ones(n_doors) / combination_number(n_doors-2,len(opened))
        likelihood_given_hi[chosen] = 1 / combination_number(n_doors-1,len(opened))
        for open_gate in opened:
            likelihood_given_hi[open_gate] = 0
    
        normalizing_constant = np.sum(prior * likelihood_given_hi)

        posterior = [pri * lk / normalizing_constant for pri, lk in zip(prior, likelihood_given_hi)]
    elif host == "random":
         num_opened = len(opened)
         num_not_opened = n_doors - num_opened
         posterior = np.ones(n_doors) / num_not_opened
         posterior[opened] = 0
         posterior = posterior.tolist()
    
    return [float(pos) for pos in posterior]
    




