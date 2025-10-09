import functools

import jax
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from diffusion_trajopt.diffusion_opt import DiffusionOptimiser
from diffusion_trajopt.trajopt import DiffusionTrajOpt, DiffusionTrajOptProjection, rollout_env
from diffusion_trajopt.environments.obs2d_navigator import (
    ObstacleNavigator,
    render_trajectory,
    render_multiple_trajectories,
)
from diffusion_trajopt.projection import project_to_obstacle_avoiding_space


from jax import config
from jax.random import PRNGKey

config.update("jax_enable_x64", True)


def graph_trajectory_distribution(trajs, name):
    render_multiple_trajectories(
        trajs,
        obstacles=sys.get_obstacles(),
        target=sys.target_position,
        colors=None,  # Use default colors
        labels=None,  # Use default labels
        title="Trajectory Distribution",
        figsize=(6, 4),
    )
    plt.savefig(
        f"{name}",
        dpi=300,
        bbox_inches="tight",
    )


def run_vectorized_trajectory_optimization(trajopt, state_init, num_trajectories, **kwargs):
    """Run trajectory optimization for multiple trajectories using JAX JIT and vmap."""
    import time
    
    # Create seeds for all trajectories
    seeds = jnp.arange(num_trajectories)
    
    # Create the vectorized optimization function
    def optimize_multiple_trajectories(state, seeds):
        return jax.vmap(
            lambda seed: trajopt.optimise_trajectory(state, seed, **kwargs)
        )(seeds)
    
    # JIT compile the function
    jitted_optimize = jax.jit(optimize_multiple_trajectories)
    
    print(f"JIT compiling for {num_trajectories} trajectories...")
    
    # First call to trigger JIT compilation (warm-up)
    _ = jitted_optimize(state_init, seeds)
    
    print(f"Running {num_trajectories} trajectories...")
    
    # Second call for actual execution (JIT already compiled) - time this
    start_time = time.time()
    actions = jitted_optimize(state_init, seeds)
    end_time = time.time()
    
    execution_time = end_time - start_time
    avg_time_per_trajectory = execution_time / num_trajectories
    
    print(f"  Execution time: {execution_time:.4f}s total, {avg_time_per_trajectory:.6f}s per trajectory")
    
    return actions, execution_time, avg_time_per_trajectory


if __name__ == "__main__":
    rng = jax.random.key(1)

    obstacle_config = {
        "radius": 0.80,
    }
    kwargs = {
        "step_size": 0.30,
        "obstacles": True,
        "target_position": jnp.array([5.0, 3.0]),
        "obstacle_config": obstacle_config,
    }
    sys = ObstacleNavigator(rng, **kwargs)
    opt = DiffusionOptimiser(
        temperature=0.01, optimisation_steps=100, sample_size=350)

    step = jax.jit(lambda s, a: sys.step(s, a))
    reset_env = jax.jit(lambda r: sys.reset(r))

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)
    rollout = jax.jit(functools.partial(rollout_env, sys, state_init))

    # Use DiffusionTrajOptMJX for single trajectory optimization with emerging barriers
    trajopt_project = DiffusionTrajOptProjection(opt, sys, 70)

    trajopt_emerging = DiffusionTrajOpt(opt, sys, 70, mu=1)
    normalising_factor = 5

    # # Create a simple random trajectory for comparison (before optimization)
    rng, rng_random = jax.random.split(rng)
    random_actions = jax.random.normal(
        rng_random, shape=(70, sys.action_size)) + 0.2
    random_states, _, _ = rollout(random_actions)

    # Project the random actions to the obstacle manifold
    projected_actions = project_to_obstacle_avoiding_space(
        rollout, random_actions)
    projected_states, _, _ = rollout(projected_actions)

    def project_actions_relaxed(actions, prog): return functools.partial(
        project_to_obstacle_avoiding_space, rollout)(actions.reshape(trajopt_project.opt_state_shape), prog).flatten()

    def project_actions(actions, prog): return functools.partial(
        project_to_obstacle_avoiding_space, rollout)(actions.reshape(trajopt_project.opt_state_shape)).flatten()

    # Number of trajectories to optimize
    num_trajectories = 1
    
    print("Running vectorized trajectory optimization...")
    
    vmapped_rollout = jax.vmap(rollout)
    total_cost = lambda actions: jnp.mean(jnp.sum(vmapped_rollout(actions)[1], axis=1) + vmapped_rollout(actions)[2])
    avg_term_dist = lambda actions: jnp.mean(
        jnp.linalg.norm(sys.target_position - vmapped_rollout(actions)[0].position[:,-1,:], axis=1)
    )
    # Run vectorized optimization for emerging barrier method
    actions_emerging, exec_time_emerging, avg_time_emerging = run_vectorized_trajectory_optimization(
        trajopt_emerging,
        state_init,
        num_trajectories,
        emerging_barrier=True,
        normalising_factor=normalising_factor,
        alpha=0.31,
        violation_higher_bound=0.8,
    )
    avg_costs_prog = total_cost(actions_emerging)
    avg_terminal_dist_prog = avg_term_dist(actions_emerging)

    # Run vectorized optimization for naive method
    actions_naive, exec_time_naive, avg_time_naive = run_vectorized_trajectory_optimization(
        trajopt_emerging,
        state_init,
        num_trajectories,
        emerging_barrier=False,
        normalising_factor=normalising_factor,
    )
    avg_costs_naive = total_cost(actions_naive)
    avg_terminal_dist_naive = avg_term_dist(actions_naive)

    # Run vectorized optimization for DPCC method
    actions_dpcc, exec_time_dpcc, avg_time_dpcc = run_vectorized_trajectory_optimization(
        trajopt_project,
        state_init,
        num_trajectories,
        normalising_factor=normalising_factor,
        projection_fn=project_actions_relaxed
    )
    avg_costs_dpcc = total_cost(actions_dpcc)
    avg_terminal_dist_dpcc = avg_term_dist(actions_dpcc)

    # Run vectorized optimization for projection method
    actions_proj, exec_time_proj, avg_time_proj = run_vectorized_trajectory_optimization(
        trajopt_project,
        state_init,
        num_trajectories,
        normalising_factor=normalising_factor,
        projection_fn=project_actions
    )
    avg_costs_proj = total_cost(actions_proj)
    avg_terminal_dist_proj = avg_term_dist(actions_proj)
    # Rollout all optimized trajectories
    print("Rolling out trajectories...")
    # Rollout for all seeds
    print(f"Average costs (emerging): {avg_costs_prog:.4f}")
    print(f"Average costs (DPCC): {avg_costs_dpcc:.4f}")
    print(f"Average costs (Projection): {avg_costs_proj:.4f}")
    print(f"Average costs (Naive): {avg_costs_naive:.4f}")

    print(f"Average Terminal Distance (emerging): {avg_terminal_dist_prog:.4f}")
    print(f"Average Terminal Distance (DPCC): {avg_terminal_dist_dpcc:.4f}")
    print(f"Average Terminal Distance (Projection): {avg_terminal_dist_proj:.4f}")
    print(f"Average Terminal Distance (Naive): {avg_terminal_dist_naive:.4f}")


    # Rollout all trajectories for each method for a single seed
    states_dpcc = rollout(actions_dpcc[0])[0]
    states_proj = rollout(actions_proj[0])[0]
    states_emerging = rollout(actions_emerging[0])[0]
    states_naive = rollout(actions_naive[0])[0]

    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY:")
    print("=" * 60)
    print(f"emerging Barrier:     {exec_time_emerging:.4f}s total, {avg_time_emerging:.6f}s per trajectory")
    print(f"Naive MBD:               {exec_time_naive:.4f}s total, {avg_time_naive:.6f}s per trajectory")
    print(f"DPCC (Relaxed Proj):     {exec_time_dpcc:.4f}s total, {avg_time_dpcc:.6f}s per trajectory")
    print(f"Projection Based:        {exec_time_proj:.4f}s total, {avg_time_proj:.6f}s per trajectory")
    
    # Find fastest and slowest methods
    timing_data = {
        'emerging Barrier': avg_time_emerging,
        'Naive MBD': avg_time_naive,
        'DPCC (Relaxed Proj)': avg_time_dpcc,
        'Projection Based': avg_time_proj
    }
    
    fastest = min(timing_data.items(), key=lambda x: x[1])
    slowest = max(timing_data.items(), key=lambda x: x[1])
    
    print(f"\nFastest method: {fastest[0]} ({fastest[1]:.6f}s per trajectory)")
    print(f"Slowest method: {slowest[0]} ({slowest[1]:.6f}s per trajectory)")
    
    if fastest[0] != slowest[0]:
        speedup = slowest[1] / fastest[1]
        print(f"Speedup: {speedup:.2f}x faster than slowest method")
    
    print("=" * 60)

    stack_state_trajs = lambda *states: jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *states)
    
    render_multiple_trajectories(
        stack_state_trajs(states_naive,
                       states_proj,
                       states_dpcc,
                       states_emerging),
        target=sys.target_position,
        obstacles=sys.get_obstacles(),
        labels=["MBD",
                "Projected MBD",
                "DPCC-MBD",
                "EB-MBD (Ours)"],
        colors=["black", "gray", "blue", "green", "red", "green"],
        figsize=(6, 4),
        title="Comparison of various MBD variants",
        save_path="projectionBeforeAfter.pdf"
    )
