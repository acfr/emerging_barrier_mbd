from jax.random import PRNGKey
from jax import config
import functools

import jax

import jax.numpy as jnp
from random import randint
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import scienceplots
from matplotlib import cm
import matplotlib.pyplot as plt
plt.style.use(['science', 'ieee', 'std-colors'])
import numpy as np
plt.rcParams.update({'font.size':14})

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
        colors=['green']*len(trajs.position),  # Use default colors
        labels=None,  # Use default labels
        title="Trajectory Distribution",
        traj_alpha=0.20,
        figsize=(5, 4),
        horizontal=True
    )
    plt.savefig(
        f"./{name}",
        dpi=30,
        bbox_inches="tight",
    )


def optimize_and_rollout(trajopt, state_init, start_seed, num, emerging_barrier=True, **kwargs):
    """Helper function to optimize trajectories and get rollouts."""
    actionss = jax.vmap(
        lambda *x: trajopt.optimise_trajectory(
            *x, emerging_barrier=emerging_barrier, **kwargs),
        in_axes=(None, 0)
    )(state_init, jnp.arange(start_seed, start_seed + num))
    rollouts = jax.vmap(rollout, in_axes=(None, 0))(state_init, actionss)
    return rollouts[0]


def run_trajectory_comparison(state_init, start_seed, num, sys, sys_lc, opt, horizon):
    """Run trajectory comparison with different optimization settings."""
    
    # Configuration for different experiments
    experiments = [
        {
            'name': 'Figure_Successful_Trajectory.pdf',
            'sys': sys,
            'trajopt': DiffusionTrajOpt(opt, sys, horizon, mu=10),
            'kwargs': {
                'emerging_barrier': True,
                'normalising_factor': 5,
                'violation_higher_bound': 0.80,
                'alpha': 0.4
            }
        },
        {
            'name': 'Figure_Failed_Trajectory.pdf',
            'sys': sys,
            'trajopt': DiffusionTrajOpt(opt, sys, horizon),
            'kwargs': {
                'emerging_barrier': False,
                'normalising_factor': 5
            }
        },
        {
            'name': 'Figure_Vanilla_MBD_LC.pdf',
            'sys': sys_lc,
            'trajopt': DiffusionTrajOpt(opt, sys_lc, horizon),
            'kwargs': {
                'emerging_barrier': False,
                'normalising_factor': 5
            }
        }
    ]

    # Run all experiments
    for exp in experiments:
        trajs = optimize_and_rollout(exp['trajopt'], state_init, start_seed, num, **exp['kwargs'])
        graph_trajectory_distribution(trajs, exp['name'], exp['sys'])


def collect_data_for_alpha(alpha, num_seeds, sample_size, temperature, optimisation_steps, horizon, mu, obstacle_config, state_init):
    """Collect data for a single alpha value across multiple seeds."""
    all_constraint_violations = []
    all_mean_costs = []

    for seed in range(num_seeds):
        # Reset the optimizer state for each seed
        opt = DiffusionOptimiser(
            temperature=temperature,
            optimisation_steps=optimisation_steps,
            sample_size=sample_size,
            store_history=True
        )
        trajopt = DiffusionTrajOpt(opt, sys, horizon, mu=mu)

        actions = trajopt.optimise_trajectory(
            state_init, seed, emerging_barrier=True,
            normalising_factor=5, alpha=alpha,
            violation_higher_bound=obstacle_config["radius"]
        )

        # rollout(state_init, actions)
        # breakpoint()

        # Extract constraint violations and costs
        constraint_violations = [
            s.constraint_violations for s in opt.state_history]
        mean_costs = [
            s.mean_cost for s in opt.state_history if s.mean_cost < 10000]

        all_constraint_violations.append(constraint_violations)
        all_mean_costs.append(mean_costs)

    # Average across seeds
    avg_constraint_violations = jnp.mean(
        jnp.array(all_constraint_violations), axis=0)

    # Average mean costs across seeds (handle different lengths)
    max_len = max(len(costs) for costs in all_mean_costs)
    padded_costs = [
        costs + [costs[-1]] * (max_len - len(costs)
                               ) if len(costs) < max_len else costs
        for costs in all_mean_costs
    ]
    avg_mean_costs = jnp.mean(jnp.array(padded_costs), axis=0)

    return avg_constraint_violations, avg_mean_costs, padded_costs[-1]


def create_plot_configs():
    """Define plot configurations."""
    return [
        {
            'name': 'constraint_violations_progress.pdf',
            'title': 'Constraint Violations Over Diffusion Process',
            'ylabel': 'Constraint Violations (\%)',
            'data_key': 'constraint_violations',
            'normalize': True
        },
        {
            'name': 'mean_costs_progress.pdf',
            'title': 'Minimum Sample Cost Over Diffusion Process',
            'ylabel': 'Minimum Sample Cost',
            'data_key': 'mean_costs',
            'normalize': False,
            'ylim': [200, 700]
        }
    ]


def run_statistics_experiment(state_init, sys, kwargs, normalising_factor):
    """Run the statistics experiment with different alpha values."""
    
    # Experiment parameters
    alphas = [0.05, 0.5, 1, 1.5, 2, 10]
    num_seeds = 15
    sample_size = 200
    temperature = 0.01
    optimisation_steps = 100
    horizon = 70
    mu = 10
    obstacle_config = kwargs['obstacle_config']
    inv_sample_size_percent = int(sample_size/100)

    # Collect data for all alphas
    all_avg_constraint_violations = []
    all_avg_mean_costs = []
    final_mean_costs = []

    for alpha in alphas:
        avg_constraint_violations, avg_mean_costs, final_cost = collect_data_for_alpha(
            alpha, num_seeds, sample_size, temperature, optimisation_steps, horizon, mu, obstacle_config, state_init
        )
        # breakpoint()
        all_avg_constraint_violations.append(avg_constraint_violations)
        all_avg_mean_costs.append(avg_mean_costs)
        final_mean_costs.append(final_cost)

    # Common plotting variables
    t = jnp.linspace(optimisation_steps, 0, optimisation_steps)
    figsize = (6, 4)
    save_path_base = "."

    # Create plots
    plot_configs = create_plot_configs()

    for config in plot_configs:
        plt.figure(figsize=figsize)

        for i, alpha in enumerate(alphas):
            data = all_avg_constraint_violations[i] if config[
                'data_key'] == 'constraint_violations' else all_avg_mean_costs[i]
            normalize_factor = inv_sample_size_percent if config.get(
                'normalize', False) else 1
            plt.plot(t[:len(data)], data/normalize_factor, label=f'$\\kappa={alpha}$')

        plt.xlim(optimisation_steps, 0)
        plt.xlabel('Diffusion Process Step (s)')
        plt.ylabel(config['ylabel'])
        # plt.title(config['title'])
        plt.legend()
        if 'ylim' in config:
            plt.ylim(config['ylim'])
        plt.grid(True, alpha=0.3)
        plt.savefig(
            f"{save_path_base}{config['name']}", dpi=300, bbox_inches="tight")
        plt.close()

    # Scatter plot of final mean costs
    plt.figure(figsize=(8/1.5, 6/1.5))

    def flatten(li):
        return [item for row in li for item in row]

    corresponding_alphas = flatten(
        [[alpha]*len(final_mean_costs[i]) for i, alpha in enumerate(alphas)])
    plt.scatter(corresponding_alphas, flatten(final_mean_costs), s=100, alpha=0.7,
                c=[cm.viridis(alphas[0] / max(alphas)) for alpha in corresponding_alphas])
    plt.xlabel('Alpha')
    plt.ylabel('Cost')
    plt.title('Minimum Sample Augmented Cost (Averaged over 10 Seeds)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path_base}final_min_costs_vs_alpha.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()

    # Unconstrained optimization plot
    opt = DiffusionOptimiser(temperature=0.01, optimisation_steps=100, sample_size=200)
    trajopt = DiffusionTrajOpt(opt, sys, 70, mu=10)
    actions = trajopt.optimise_trajectory(
        state_init, 1, emerging_barrier=False, normalising_factor=normalising_factor)
    plt.clf()
    plt.plot([s.constraint_violations for s in opt.state_history])
    plt.savefig(
        f"{save_path_base}constraint_violations_progress_unconstrained.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=[
                        'statistics', 'comparison', 'time'])
    args = parser.parse_args()

    rng = jax.random.key(100)

    kwargs = {
        "step_size": 0.30,
        "obstacles": True,
        "target_position": jnp.array([3.0, 5.0]),
        "obstacle_config": {
            "radius": 0.80
        },
    }
    # Low constraint radius
    kwargs_lc = kwargs.copy()
    kwargs_lc["obstacle_config"] = {
        "radius": 0.45
    }

    horizon = 70
    sys = ObstacleNavigator(rng, **kwargs)
    sys_lc = ObstacleNavigator(rng, **kwargs_lc)
    
    opt = DiffusionOptimiser(
        temperature=0.1, optimisation_steps=100, sample_size=440)

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

    if args.command == 'comparison':
        run_trajectory_comparison(
            state_init, start_seed, num, sys, sys_lc, opt, horizon)
    elif args.command == 'statistics':
        run_statistics_experiment(state_init, sys, kwargs, normalising_factor)
