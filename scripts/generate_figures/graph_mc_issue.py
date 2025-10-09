from diffusion_trajopt.mc_gaussian_smoothing import (
    gaussian_approx,
    gaussian,
    convolve_with_time,
)
from diffusion_trajopt.constraints import log_barrier
from diffusion_trajopt.diffusion_utils import beta_schedule

import jax.numpy as jnp
import jax.random
import scienceplots
import matplotlib.pyplot as plt
import matplotlib

from typing import Tuple
from random import randint


matplotlib.use("QtAgg")

T = 1.0


def f(x):
    return x**2 + jnp.sin(10 * x) + 2


def dfdx(x):
    return jax.grad(f)(x)


def potential(x):
    return jnp.exp(-f(x) / T)


def indicator(x):
    return jnp.where(x < 1, jnp.where(x > -0.3, 0, 1), 1)


# def make_heatmap_with_approx(
#         func, approx_maker, x_range: Tuple[float, float], sigma
# ):
#     x = jnp.linspace(*x_range, 300)
#     X, Y = jnp.meshgrid(x, sigma)
#     Z = jnp.zeros_like(X)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             f = approx_maker(func, Y[i, j])
#             Z = Z.at[i, j].set(
#                 jnp.sum(f(X[i, j], jax.random.PRNGKey(randint(0, 100)), 10))
#             )
#     return X, Y, Z


def make_heatmap_with_approx(func, approx_maker, x_range: Tuple[float, float], alpha_bar, sigma):
    x = jnp.linspace(*x_range, 300)
    X, Y = jnp.meshgrid(x, sigma)
    Z = jnp.zeros_like(X)
    for i in range(X.shape[0]):
        adjusted_func = lambda x: func(x/alpha_bar[i])
        f = approx_maker(adjusted_func, Y[i, 0])
        f_val = f(X[i], jax.random.PRNGKey(randint(0, 100)), 100)
        f_val = f_val/f_val.sum()
        Z = Z.at[i].set(f_val)
    return Z.T


def make_heatmap_with_conv(func, x_range: Tuple[float, float], alpha_bar, sigma):
    x = jnp.linspace(*x_range, 300)
    X, Y = jnp.meshgrid(x, sigma)
    Z = jnp.zeros_like(X)
    for i in range(X.shape[0]):
        original_func = func(x/alpha_bar[i])
        sigma = Y[i, 0]
        alpha_bar = alpha_bar
        convolved = convolve_with_time(
            x, gaussian(sigma)(x), original_func, mode="same"
        )[1]
        f_val = convolved/convolved.sum()
        Z = Z.at[i].set(f_val)
    return Z.T


if __name__ == "__main__":
    x = jnp.linspace(-2, 2, 100)
    V_with_indicator = lambda x: potential(x) * indicator(x)
    plt.style.use(["science", "no-latex", "ieee"])
    test_sigma = 0.3
    test_function = jax.jit(V_with_indicator)
    # plt.figure(0)
    # plt.plot(*convolve_with_time(x, gaussian_derivative(test_sigma)(x), test_function(x)))
    # plt.plot(
    #     x,
    #     gaussian_derivative_approx(test_function, test_sigma)(x, jax.random.PRNGKey(0), 10),
    # )
    # plt.axvspan(-0.3, 1, color='gray', alpha=0.3)
    # plt.legend(["$V(x) * G'(x)$", "MC10 $V_{\\sigma}'(x)$"])
    # plt.xlim(-2, 2)
    #

    # plt.figure(1)
    # plt.plot(*convolve_with_time(x, gaussian_derivative(test_sigma)(x), test_function(x)))
    # plt.plot(
    #     x,
    #     gaussian_derivative_approx(test_function, test_sigma)(x, jax.random.PRNGKey(0), 100),
    # )
    # plt.axvspan(-0.3, 1, color='gray', alpha=0.3)
    # plt.legend(["$V(x) * G'(x)$", "MC100 $V_{\\sigma}'(x)$"])
    # plt.xlim(-2, 2)

    # plt.figure(2)
    # plt.plot(x, test_function(x))
    # plt.plot(*convolve_with_time(x, gaussian(test_sigma)(x), test_function(x)))
    # # plt.plot(*convolve_with_time(x, gaussian_derivative(test_sigma)(x), test_function(x)))
    # plt.plot(
    #     x,
    #     gaussian_approx(test_function, test_sigma)(x, jax.random.PRNGKey(0), 10),
    # )
    plt.figure(3)
    beta_sched = beta_schedule(1e-4, 1e-2, 300)
    alphas = 1 - beta_sched
    alpha_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alpha_bar)
    hm1 = make_heatmap_with_approx(test_function, gaussian_approx, (-2, 2), alpha_bar, sigmas)
    plt.imshow(hm1, extent=(sigmas[0], sigmas[-1], -2, 2), aspect="auto")
    plt.savefig("diffusion_heatmap_approx.pdf")
    plt.figure(4)
    hm2 = make_heatmap_with_conv(test_function, (-2, 2), alpha_bar, sigmas)
    plt.imshow(hm2, extent=(sigmas[0], sigmas[-1], -2, 2), aspect="auto")
    plt.savefig("diffusion_heatmap_analytical.pdf")
    # plt.plot(hm2[:, -1])
    # plt.show()
