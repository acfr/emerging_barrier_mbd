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
    parser.add_argument("algo", choices=["ebmbd", "mbd"])
    args = parser.parse_args()

    rng = jax.random.key(100)

    kwargs = {
        "step_size": 0.40,
        "obstacles": True,
        "target_position": jnp.array([3.0, 5.0]),
        "obstacle_config": {"radius": 0.60},
    }
    # Low constraint radius
    kwargs_lc = kwargs.copy()
    kwargs_lc["obstacle_config"] = {"radius": 0.45}

    horizon = 55
    sys = ObstacleNavigator(rng, **kwargs)
    sys_lc = ObstacleNavigator(rng, **kwargs_lc)

    opt = DiffusionOptimiser(
        temperature=0.1,
        optimisation_steps=100,
        sample_size=440,
        store_history=True,
        noise=True,
    )

    step = jax.jit(lambda s, a: sys.step(s, a))
    reset_env = jax.jit(lambda r: sys.reset(r))
    rollout = jax.jit(functools.partial(rollout_env, sys))

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    start_seed = 0
    num = 100

    # Use DiffusionTrajOptMJX directly for constrained optimization
    trajopt = DiffusionTrajOpt(opt, sys, horizon, mu=10)
    normalising_factor = 5
    barrier_args = {
        "emerging_barrier": args.algo == "ebmbd",
        "normalising_factor": normalising_factor,
        "violation_higher_bound": 0.8,
        "alpha": 0.4,
    }
    actions = jax.vmap(
        lambda *x: trajopt.optimise_trajectory(*x, **barrier_args), in_axes=(None, 0)
    )(state_init, jnp.arange(start_seed, start_seed + num))

    traj, _, _ = jax.vmap(rollout, in_axes=(None, 0))(state_init, actions)
    actions = tuple(
        trajopt._reshape_normalise_act(hist.Y_i.val, normalising_factor, True)
        for hist in trajopt.optimiser.state_history
    )
    actions_stacked = np.concatenate(actions)
    rollout_vmap = jax.vmap(rollout, in_axes=(None, 0))
    trajs = tuple(rollout_vmap(state_init, actions) for actions in actions)
    trajs_stacked = np.concatenate(tuple(traj[0].position for traj in trajs))
    costs_stacked = np.concatenate(
        tuple(np.sum(traj[1], axis=1) + traj[2] for traj in trajs)
    )

    np.savez(
        "dataset.npz",
        actions=actions_stacked,
        trajectories=trajs_stacked,
        costs=costs_stacked,
        obstacle_x_y_radius=sys.get_obstacles(),
        target_position=kwargs["target_position"],
        starting_position=state_init.position,
        cost_code="""

    def stage_cost(self, state: NavigatorState, action: jax.Array) -> jax.Array:
        # Calculate stage cost for a single step
        # Base cost is distance to target
        position_cost = jnp.linalg.norm(state.position - self.target_position)
        action_cost = jnp.linalg.norm(action) / 10

        # Add collision penalty
        sdf_value = self.sdf_fn(state)
        collision_penalty = jnp.where(sdf_value <= 0, 10.0, 0.0)[0]
        total_cost = action_cost + position_cost + collision_penalty
        return total_cost

    def terminal_cost(self, state: NavigatorState) -> jax.Array:
        # Calculate terminal cost.
        return self.stage_cost(state, jnp.array([0.0, 0.0]))*20
        """
    )
    #
    # for i, hist in enumerate(trajopt.optimiser.state_history):
    #     if i % 5 != 0:
    #         continue
    #     new_actions = trajopt._reshape_normalise_act(
    #         hist.Y_i.val, normalising_factor, True
    #     )
    #     traj, _, _ = jax.vmap(rollout, in_axes=(None, 0))(state_init, new_actions)
    #     fig, ax = render_multiple_trajectories(
    #         traj, sys.target_position, sys.get_obstacles(), horizontal=True
    #     )
    #     plt.show()
