import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the discounted return for a given list of rewards.
    Args:
      rewards (list of float): sequence of rewards R_{t+1}, R_{t+2}, ...
      gamma (float): discount factor (0 <= gamma <= 1)
    Returns:
      float: discounted return G_t
    """
    
    steps = len(rewards)
    disc_ret = 0
    for i in range(steps):
        disc_ret += rewards[i] * gamma**i 
    
    return disc_ret