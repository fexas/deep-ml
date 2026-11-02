import numpy as np

def reparameterize(mu, logvar, eps):
    """z = mu + exp(0.5 * logvar) * eps, all shapes (batch, latent_dim)."""
    z = np.exp(0.5 * logvar) * eps + mu
    return z

def kl_to_standard_normal(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) summed over latent dims, averaged over the batch. Returns float."""
    
    kl_r = 0.5 * ( mu*mu + np.exp(logvar) - 1 - logvar )
    kl_r = np.mean(np.sum(kl_r,axis = 1))

    return kl_r.item()
