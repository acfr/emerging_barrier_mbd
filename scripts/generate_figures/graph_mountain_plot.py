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

plt.style.use(["science", "no-latex"])
# Use the high-vis color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[
        # "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
)
matplotlib.rcParams.update({'font.size': 7})


T = 3.0


def f(x):
    return x**2 + jnp.sin(10 * x) + 2


def dfdx(x):
    return jax.grad(f)(x)


def V(x):
    return jnp.exp(-f(x) / T)


def indicator(x):
    return jnp.where(x < 1, jnp.where(x > -1, 0, 1), 1)


def sdf(x):
    return jnp.abs(x) - 1


def return_sdf_and_maximum_bound() -> Tuple:
    return sdf, 1


if __name__ == "__main__":
    # Plot the probability function V(x) - mu log(sdf(x))
    x = jnp.linspace(-3, 3, 1000)
    mu = 1
    # Plots
    # sdf | cost | V
    # row each for
    ts = [1.1, 0.5, 0]
    funcs = [
        lambda x, t: sdf(x) + t,
        lambda x, t: f(x) + log_barrier(sdf(x) + t, mu),
        lambda x, t: (p := (V(f(x) + log_barrier(sdf(x) + t, mu)))) / jnp.sum(p),
    ]
    ylims = [[-2, 2], [-2, 12], [0, 0.026]]
    ylabels = ["$g(x) + c(t)$", "$\\hat J(x)$", "$\\hat p_0(x)$"]
    for i in range(3):
        for j in range(3):

            t = ts[i]
            axes = plt.subplot(3, 3, j * 3 + i + 1)
            axes.get_figure().set_size_inches(4, 3)
            func = jax.vmap(funcs[j], in_axes=(0,None))
            plt.plot(x, func(x, t))
            plt.ylim(ylims[j])
            plt.plot(x, jnp.zeros_like(x), "k--")
            # plot vertical line at x=1 and x=-1
            plt.plot([1, 1], [ylims[j][0], ylims[j][1]], "k--")
            plt.plot([-1, -1], [ylims[j][0], ylims[j][1]], "k--")
            breakpoint()

            plt.tight_layout(pad=0.3)
            if j != 2:
                plt.xticks([])
            if i == 0:
                plt.ylabel(ylabels[j])
            else:
                plt.yticks([])

            if j == 0:
                plt.title(f"$c(t)={t}$")
            plt.subplots_adjust(wspace=0.1)

    plt.savefig("mountain_plot.pdf")
    # plt.show()
