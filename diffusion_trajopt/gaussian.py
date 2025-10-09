import jax.numpy as np

def halfspace_gaussian_cdf(w, b, mu, sigma):
    """
    Finds the definite integral of the multivariate gaussian described by
    N(mu, sigma) over the halfspace defined by w^T x + b >= 0.

    Source: https://math.stackexchange.com/questions/556977/gaussian-integrals-over-a-half-space
    """
    # Find cholensky decomposition of sigma
    L = np.linalg.cholesky(sigma)
