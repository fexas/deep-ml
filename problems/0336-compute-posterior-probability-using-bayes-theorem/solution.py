def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.
    
    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis
        
    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """

    evidence = 0

    for prior, likelihood in zip(priors, likelihoods):
        evidence += prior * likelihood
    
    return [prior * likelihood / evidence for prior, likelihood in zip(priors, likelihoods)]
