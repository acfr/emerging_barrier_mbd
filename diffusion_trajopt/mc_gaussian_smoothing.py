import jax.numpy as jnp
import jax


def gaussian_approx(f, sigma):
    # The sigma of which u should be sampled from
    sigma_standard = 1 / jnp.sqrt(2)

    def f_sigma(x, rng, N):
        u = jax.random.normal(rng, (N, len(x))) * sigma_standard
        vf = jax.vmap(f)
        perturbed_xs = jnp.expand_dims(x, 0) + u * sigma
        return 1 / N * jnp.sum(vf(perturbed_xs), axis=0)

    return f_sigma


def gaussian_derivative_approx(f, sigma):
    # The sigma of which u should be sampled from
    sigma_standard = 1 / jnp.sqrt(2)

    def f_sigma(x, rng, N):
        u = jax.random.normal(rng, (N,)) * sigma_standard
        vf = jax.vmap(f)
        perturbed_xs = jnp.expand_dims(x, 0) + jnp.expand_dims(u, 1) * sigma
        delta_sigma = (vf(perturbed_xs) - f(x)) / (sigma)
        delta_sigma = delta_sigma * u[:, None]
        return 1 / N * jnp.sum(delta_sigma, axis=0)

    return f_sigma


def gaussian(sigma):
    return (
        lambda x: jnp.exp(-(x**2) / (1 * sigma**2)) * 1 / (jnp.sqrt(2 * jnp.pi) * sigma)
    )


def gaussian_derivative(sigma):
    gaussian_f = gaussian(sigma)
    return lambda x: (-x / (sigma**2)) * gaussian_f(x)


def convolve_with_time(x, f_x, g_x, mode="full"):
    """
    Convolve two signals and return both the new time axis and the convolution result.

    Parameters:
    -----------
    x : numpy.ndarray
        Original time/space axis for both signals
    f_x : numpy.ndarray
        First signal values
    g_x : numpy.ndarray
        Second signal values
    mode : str, optional
        Convolution mode ('full', 'same', or 'valid'), default is 'full'

    Returns:
    --------
    x_conv : numpy.ndarray
        Time/space axis for the convolution result
    conv_result : numpy.ndarray
        Convolution result (f_x * g_x)
    """
    # Check input dimensions
    if len(f_x) != len(x) or len(g_x) != len(x):
        raise ValueError("Input signals must have the same length as the time axis")

    # Get step size from original time axis
    dx = x[1] - x[0]

    # Perform convolution
    conv_result = jnp.convolve(f_x, g_x, mode=mode)

    # Create corresponding time axis based on mode
    if mode == "full":
        # For 'full' mode: Output length is len(f_x) + len(g_x) - 1
        x_min = 2 * x[0]
        x_max = 2 * x[-1]
        x_conv = jnp.linspace(x_min, x_max, len(conv_result))

    elif mode == "same":
        # For 'same' mode: Output length is max(len(f_x), len(g_x))
        x_conv = x.copy()  # Same time axis as input

    else:
        # For 'valid' mode: Output length is max(len(f_x), len(g_x)) - min(len(f_x), len(g_x)) + 1
        offset = min(len(f_x), len(g_x)) - 1
        x_conv = x[offset // 2 : -(offset // 2)] if offset > 0 else x

    # Scale the convolution result by dx to maintain physical units
    conv_result *= dx

    return x_conv, conv_result
