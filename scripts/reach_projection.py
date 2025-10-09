import functools

import jax
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from diffusion_trajopt.diffusion_opt import DiffusionOptimiser
from diffusion_trajopt.trajopt import DiffusionTrajOpt, DiffusionTrajOptProjection, rollout_env
from reach_mjx.alpha_bluerov_brax_env import ReachArm

from diffusion_trajopt.environments.obs2d_navigator import (
    render_trajectory,
    render_multiple_trajectories,
)
from diffusion_trajopt.projection import project_to_obstacle_avoiding_space


from jax import config
from jax.random import PRNGKey

config.update("jax_enable_x64", True)


def run_vectorized_trajectory_optimization(trajopt, state_init, **kwargs):
    """Run trajectory optimization for multiple trajectories using JAX JIT and vmap."""
    import time

    seed = 0

    # Create the vectorized optimization function
    def optimize_trajectory(state):
        return trajopt.optimise_trajectory(state, seed, **kwargs)

    # JIT compile the function
    jitted_optimize = (optimize_trajectory)

    print(f"JIT compiling for {num_trajectories} trajectories...")

    # First call to trigger JIT compilation (warm-up)
    # _ = jitted_optimize(state_init)

    print(f"Running {num_trajectories} trajectories...")

    # Second call for actual execution (JIT already compiled) - time this
    start_time = time.time()
    actions = jitted_optimize(state_init)
    end_time = time.time()

    execution_time = end_time - start_time
    print(
        f"  Execution time: {execution_time:.4f}s total")

    return actions, execution_time


if __name__ == "__main__":
    rng = jax.random.key(1)

    sys = ReachArm(rng, target_pos=jnp.array([0.5, 0.0, -0.0]))
    opt = DiffusionOptimiser(
        temperature=0.1, optimisation_steps=110, sample_size=600)

    step = jax.jit(lambda s, a: sys.step(s, a))
    reset_env = jax.jit(lambda r: sys.reset(r))

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)
    rollout = jax.jit(functools.partial(rollout_env, sys, state_init))

    # Use DiffusionTrajOptMJX for single trajectory optimization with emerging barriers
    trajopt_project = DiffusionTrajOptProjection(opt, sys, 70)

    trajopt_emerging = DiffusionTrajOpt(opt, sys, 70, mu=1)
    normalising_factor = 10

    def project_actions_relaxed(actions, prog): return functools.partial(
        project_to_obstacle_avoiding_space, rollout)(actions.reshape(trajopt_project.opt_state_shape), prog).flatten()

    def project_actions(actions, prog): return functools.partial(
        project_to_obstacle_avoiding_space, rollout)(actions.reshape(trajopt_project.opt_state_shape)).flatten()

    # Number of trajectories to optimize
    num_trajectories = 1

    print("Running vectorized trajectory optimization...")

    # Run vectorized optimization for DPCC method
    actions_dpcc, exec_time_dpcc = run_vectorized_trajectory_optimization(
        trajopt_project,
        state_init,
        normalising_factor=normalising_factor,
        projection_fn=project_actions_relaxed
    )

    # Run vectorized optimization for emerging barrier method
    actions_emerging, exec_time_emerging = run_vectorized_trajectory_optimization(
        trajopt_emerging,
        state_init,
        emerging_barrier=True,
        normalising_factor=normalising_factor,
        alpha=1.2,
        violation_higher_bound=0.08,
    )

    # Run vectorized optimization for naive method
    actions_naive, exec_time_naive = run_vectorized_trajectory_optimization(
        trajopt_emerging,
        state_init,
        emerging_barrier=False,
        normalising_factor=normalising_factor,
    )
    # Run vectorized optimization for projection method
    actions_proj, exec_time_proj = run_vectorized_trajectory_optimization(
        trajopt_project,
        state_init,
        normalising_factor=normalising_factor,
        projection_fn=project_actions
    )

    # Rollout all optimized trajectories
    print("Rolling out trajectories...")
    # Rollout for all seeds
    vmapped_rollout = jax.vmap(rollout)
    avg_costs_dpcc = jnp.mean(vmapped_rollout(actions_dpcc)[1])
    avg_costs_proj = jnp.mean(vmapped_rollout(actions_proj)[1])
    avg_costs_prog = jnp.mean(vmapped_rollout(actions_emerging)[1])
    avg_costs_naive = jnp.mean(vmapped_rollout(actions_naive)[1])
    print(f"Average costs (DPCC): {avg_costs_dpcc:.4f}")
    print(f"Average costs (Projection): {avg_costs_proj:.4f}")
    print(f"Average costs (emerging): {avg_costs_prog:.4f}")
    print(f"Average costs (Naive): {avg_costs_naive:.4f}")

    # Rollout all trajectories for each method for a single seed
    states_dpcc = rollout(actions_dpcc[0])
    states_proj = rollout(actions_proj[0])
    states_emerging = rollout(actions_emerging[0])
    states_naive = rollout(actions_naive[0])

    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY:")
    print("=" * 60)
    print(
        f"Emerging Barrier:     {exec_time_emerging:.4f}s total")
    print(
        f"Naive MBD:               {exec_time_naive:.4f}s total")
    print(
        f"DPCC (Relaxed Proj):     {exec_time_dpcc:.4f}s total")
    print(
        f"Projection Based:        {exec_time_proj:.4f}s total")

    # Find fastest and slowest methods
    timing_data = {
        'Emerging Barrier': exec_time_emerging,
        'Naive MBD': exec_time_naive,
        'DPCC (Relaxed Proj)': exec_time_dpcc,
        'Projection Based': exec_time_proj
    }

    fastest = min(timing_data.items(), key=lambda x: x[1])
    slowest = max(timing_data.items(), key=lambda x: x[1])

    print(f"\nFastest method: {fastest[0]} ({fastest[1]:.6f}s per trajectory)")
    print(f"Slowest method: {slowest[0]} ({slowest[1]:.6f}s per trajectory)")

    if fastest[0] != slowest[0]:
        speedup = slowest[1] / fastest[1]
        print(f"Speedup: {speedup:.2f}x faster than slowest method")

    print("=" * 60)
