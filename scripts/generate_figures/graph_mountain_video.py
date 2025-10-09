from diffusion_trajopt.mc_gaussian_smoothing import (
    gaussian_approx,
    gaussian,
    convolve_with_time,
)
# from diffusion_trajopt.constraints import log_barrier
from diffusion_trajopt.diffusion_utils import beta_schedule

from matplotlib.animation import FuncAnimation

import jax.numpy as jnp
import jax.random
import scienceplots
import matplotlib.pyplot as plt
import matplotlib

np = jnp

from typing import Tuple
from random import randint

# matplotlib.use("QtAgg")

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
# matplotlib.rcParams.update({'font.size': 7})

T = 10


def f(x):
    return x**2 + jnp.sin(10 * x) + 2


def dfdx(x):
    return jax.grad(f)(x)


def boltz(x):
    return jnp.exp(-f(x) / T)


def indicator(x):
    return jnp.where(x < 1, jnp.where(x > -1, 0, 1), 1)


def sdf(x):
    return jnp.abs(x) - 1


def log_barrier(g, mu):
    """Log barrier function: -mu * log(-g) for g < 0"""
    return jnp.where(g <= 0, -mu * jnp.log(-g), 1000)


def return_sdf_and_maximum_bound() -> Tuple:
    return sdf, 1


# Set up the figure and subplot grid
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

# Initialize the plots
x = jnp.linspace(-3, 3, 1000)
mu = 1


funcs = [
    lambda x, t: sdf(x) + t,
    lambda x, t: f(x) + log_barrier(-sdf(x) - t, mu),
    lambda x, t: boltz(f(x) + log_barrier(-sdf(x) - t, mu))
]

ylims = [[-2, 2], [-2, 12], [0, 0.016]]
ylabels = ["$g(x) + c_s$", "$\\hat J(x, s)$", "$\\hat p_0(x, s)$"]
titles = ["Constraint function", "Cost Function", "Target Distribution"]

# Initialize empty lines for each plot
lines = []
# for i, ax in enumerate(axes):
ax = axes
i = 2
line, = ax.plot([], [], 'b-')
lines.append(line)

ax.set_xlim(-3, 3)
ax.set_ylim(ylims[i])
ax.set_ylabel(ylabels[i])
ax.set_title(titles[i])

# Add horizontal zero line and vertical constraint lines
ax.plot(x, jnp.zeros_like(x), "k--", alpha=0.5)
ax.plot([1, 1], [ylims[i][0], ylims[i][1]], "k--", alpha=0.5)
ax.plot([-1, -1], [ylims[i][0], ylims[i][1]], "k--", alpha=0.5)

# Add x-axis label to bottom plot
ax.set_xlabel("$x$")

# Time display
time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12)
n_frames = 500
hold_frames = 0  # Hold for 10% of total frames at each end

def update(frame):
    # Hold on first frame, animate, then hold on last frame
    # if frame < hold_frames:  # Hold on first frame
    #     t = 1.5*(1 - (0) / (n_frames - 2 * hold_frames - 5)) * 1 
    t = 1.5*(1 - (frame - 1) / (n_frames)) 
    # else:  # Hold on last frame
    #     frame = n_frames - hold_frames
    #     t = 1.5*(1 - (frame - hold_frames) / (n_frames - 2 * hold_frames - 5)) * 0.9
    # 
    time_text.set_text(f'$c(s) = {t:.2f}$')
    
    line = lines[0]
    # for i, line in enumerate(lines):
    func = jax.vmap(funcs[i], in_axes=(0, None))
    y_values = np.array(func(x, t))
    if i == 2:
        # Normalize the probability distribution
        y_values = y_values / jnp.sum(y_values)
    # print(y_values)
    # breakpoint()
    
    
    line.set_data(x, y_values)


# update(10)

# Create the animation
ani = FuncAnimation(fig, update, frames=n_frames, interval=50)

# Save the animation (uncomment to save)
plt.show()
ani.save(f"barrier_function_animation_{i}.gif", writer="pillow", fps=30, dpi=300)


