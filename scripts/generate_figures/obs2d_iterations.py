from jax.random import PRNGKey
from jax import config
import functools

import jax

import jax.numpy as jnp
from random import randint
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# import scienceplots
from matplotlib import cm
import matplotlib.pyplot as plt

# plt.style.use(["science", "ieee", "std-colors"])
import numpy as np

plt.rcParams.update({"font.size": 14})

from matplotloom import Loom
from diffusion_trajopt.diffusion_opt import DiffusionOptimiser

# from diffusion_trajopt.utils import rollout_us_arr
from diffusion_trajopt.trajopt import DiffusionTrajOpt, rollout_env
from diffusion_trajopt.environments.obs2d_navigator import (
    ObstacleNavigator,
    render_trajectory,
    render_multiple_trajectories,
)

import argparse

config.update("jax_enable_x64", True)


def graph_trajectory_distribution(trajs, name, sys):
    """Render and save trajectory distribution plots."""
    render_multiple_trajectories(
        trajs,
        obstacles=sys.get_obstacles(),
        target=sys.target_position,
        colors=["green"] * len(trajs.position),  # Use default colors
        labels=None,  # Use default labels
        title="Trajectory Distribution",
        traj_alpha=0.20,
        figsize=(5, 4),
        horizontal=True,
    )
    plt.savefig(
        f"{name}",
        dpi=30,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["samples", "seeds"])
    parser.add_argument("algo", choices=["ebmbd", "mbd"])
    args = parser.parse_args()

    rng = jax.random.key(100)

    kwargs = {
        "step_size": 0.40,
        "obstacles": True,
        "target_position": jnp.array([3.0, 5.0]),
        "obstacle_config": {"radius": 0.30},
    }
    # Low constraint radius
    kwargs_lc = kwargs.copy()
    kwargs_lc["obstacle_config"] = {"radius": 0.45}

    horizon = 70
    sys = ObstacleNavigator(rng, **kwargs)
    sys_lc = ObstacleNavigator(rng, **kwargs_lc)

    opt = DiffusionOptimiser(
        temperature=0.1, optimisation_steps=100, sample_size=440, store_history=True
    )

    step = jax.jit(lambda s, a: sys.step(s, a))
    reset_env = jax.jit(lambda r: sys.reset(r))
    rollout = jax.jit(functools.partial(rollout_env, sys))

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    start_seed = 0
    num = 50

    # Use DiffusionTrajOptMJX directly for constrained optimization
    trajopt = DiffusionTrajOpt(opt, sys, horizon, mu=10)
    normalising_factor = 5
    loom = Loom(f"obs2d_{args.algo}_{args.command}_0.3.gif", fps=10, verbose=True, overwrite=True)
    barrier_args = {
        "emerging_barrier" : args.algo == "ebmbd",
        "normalising_factor" : normalising_factor,
        "violation_higher_bound" : 0.8,
        "alpha" : 0.4,
    }
    if args.command == "samples":
        actions = trajopt.optimise_trajectory(
            state_init,
            seed=2,
            **barrier_args,
        )
    else:
        actions = jax.vmap(
            lambda *x: trajopt.optimise_trajectory(
                *x, **barrier_args),
            in_axes=(None, 0)
        )(state_init, jnp.arange(start_seed, start_seed + num))

    with loom:
        for diffusion_state in trajopt.optimiser.state_history:
            actions_flat = diffusion_state.Y_i.val if args.command == "seeds" else diffusion_state.samples
            actions = (
                actions_flat.reshape((-1,) + trajopt.opt_state_shape)
                / normalising_factor
            )
            traj, _, _ = jax.vmap(rollout, in_axes=(None, 0))(state_init, actions)
            fig, ax = render_multiple_trajectories(
                traj, sys.target_position, sys.get_obstacles(), horizontal=True
            )
            loom.save_frame(fig)
