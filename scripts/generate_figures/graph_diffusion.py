from diffusion_trajopt.mc_gaussian_smoothing import (
    gaussian_approx,
    gaussian,
    convolve_with_time,
)
from diffusion_trajopt.constraints import log_barrier
from diffusion_trajopt.diffusion_utils import beta_schedule

import scienceplots 
import matplotlib.pyplot as plt
import matplotlib

import jax
import jax.numpy as jnp


def f(x):
    return x**2 + jnp.sin(7 * x) + 2


T = 2


def potential(x):
    return jnp.exp(-f(x) / T)


def indicator(x):
    return jnp.where(x < 1, jnp.where(x > -1, 0, 1), 1)


if __name__ == "__main__":
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"figure.dpi": "300"})

    x = jnp.linspace(-5, 5, 1000)
    V_with_indicator = lambda x: potential(x) * indicator(x)
    test_function = jax.jit(V_with_indicator)

    sigmas = [0.2, 0.6, 1, 3, 8]
    alphas = [0.7, 0.55, 0.5, 0.4, 0.3]

    plt.figure(0)
    cwt = convolve_with_time(x, gaussian(1.0)(x), test_function(x))

    plt.plot(cwt[0], (p:=test_function(cwt[0]))/jnp.sum(p), c=(1.0, 0, 0, 1))
    for i in range(0, 5):
        x, y = convolve_with_time(x, gaussian(sigmas[i])(x), test_function(x))
        plt.plot(
            x, y/jnp.sum(y),
            c=(1.0, 0, 0, alphas[i]),
        )

    plt.legend(
        [
            "Original",
            f"$\\sigma$={sigmas[0]}",
            f"$\\sigma$={sigmas[1]}",
            f"$\\sigma$={sigmas[2]}",
            f"$\\sigma$={sigmas[3]}",
            f"$\\sigma$={sigmas[4]}",
        ]
    )
    plt.ylim(0, 0.018)
    plt.xlim(-5, 5)

    plt.savefig("gaussian_approx_analytical_bigger.pdf")

    # x = cwt[0]
    # plt.figure(1)
    # plt.plot(x, test_function(x), c=(1.0, 0, 0, 1))
    # for i in range(0, 4):
    #     plt.plot(
    #         x,
    #         gaussian_approx(test_function, sigmas[i])(x, jax.random.PRNGKey(i), 70),
    #         c=(1.0, 0, 0, alphas[i]),
    #     )
    # plt.ylim(0, 0.5)

    # plt.legend(
    #     [
    #         "Original",
    #         f"$\\sigma$={sigmas[0]}",
    #         f"$\\sigma$={sigmas[1]}",
    #         f"$\\sigma$={sigmas[2]}",
    #         f"$\\sigma$={sigmas[3]}",
    #     ]
    # )
