import numpy as np

def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: np.ndarray of shape (batch_size, features), original input
        x_reconstructed: np.ndarray of shape (batch_size, features), reconstructed input
        mu: np.ndarray of shape (batch_size, latent_dim), latent mean
        log_var: np.ndarray of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    # Reconstruction loss 
    recon =  np.sum( (x-x_reconstructed) * (x-x_reconstructed), axis=1)
    recon = np.mean(recon)

    # KL Regualizer
    kl_r = 0.5 * (mu * mu + np.exp(log_var)  - log_var - 1)
    kl_r = np.mean( np.sum(kl_r,axis=1) )
    
    vae_loss = recon + kl_r

    return vae_loss.item(), recon.item(), kl_r.item()
    