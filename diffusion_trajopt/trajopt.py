import functools

import jax
import jax.numpy as jnp

from .diffusion_opt import DiffusionOptimiser
from .emerging_barrier import emerging_barrier_cost, EmergingBarrierParams


@jax.jit
def rollout_env(env, state_init, actions):
    def step(state, u):
        state = env.step(state, u)
        return state, (state, env.stage_cost(state, u))

    terminal_state, (states, stage_costs) = jax.lax.scan(
        step, state_init, actions[:-1])
    states = jax.tree.map(
        lambda x, y: jnp.concatenate((jnp.expand_dims(x, axis=0), y), axis=0), state_init, states)

    return states, stage_costs, env.terminal_cost(env.step(terminal_state, actions[-1]))


def direct_transcription_cost(rollout_fn, state_init, actions):
    states, stage_costs, terminal_cost = rollout_fn(state_init, actions)
    dists: jax.Array = states.contact.dist
    cost = jnp.sum(stage_costs) + terminal_cost
    cost = jnp.where((dists < 0).any(), jnp.inf, cost)
    return cost


class DiffusionTrajOpt:
    def __init__(self, diff_optimiser: DiffusionOptimiser, env, horizon_length=300, mu=0.1):
        self.optimiser = diff_optimiser
        self.N_horizon = horizon_length
        self.action_dim = env.action_size
        self.opt_state_shape = (self.N_horizon, self.action_dim)
        self.env = env
        self.mu = mu

    def _reshape_normalise_act(self, actions, factor, batch=False):
        if not batch:
            return actions.reshape(self.opt_state_shape)/factor
        return actions.reshape((-1, ) + self.opt_state_shape)/factor

    def optimise_trajectory(self, init_state, seed, violation_higher_bound=None, alpha=1.0, normalising_factor=1.0, emerging_barrier=False, return_full=False):
        rollout = jax.jit(functools.partial(rollout_env, self.env))
        if not emerging_barrier:
            print("Using naive direct transcription cost")
            cost_func = jax.jit(
                lambda inputs, p: functools.partial(
                    direct_transcription_cost,
                    rollout,
                    init_state,
                )(self._reshape_normalise_act(inputs, normalising_factor))
            )
        elif violation_higher_bound is not None:
            print(
                f"Using emerging barriers with parameter {self.mu=}, {alpha=}, {violation_higher_bound=}")

            params = EmergingBarrierParams(
                mu=self.mu, alpha=alpha, higher_bound=violation_higher_bound, end_time=1.0)

            @jax.jit
            def barrier_rollout_cost(inputs, prog):
                actions = inputs.reshape(
                    self.opt_state_shape)/normalising_factor
                states, stage_costs, terminal_cost = rollout(
                    init_state, actions)
                traj_cost = jnp.sum(stage_costs) + terminal_cost
                barrier_cost = emerging_barrier_cost(
                    jnp.min(states.contact.dist), params, prog)
                return traj_cost + barrier_cost

            cost_func = barrier_rollout_cost
        else:
            raise ValueError("No violation higher bound provided")

        return self.optimiser.reverse_process(
            cost_func, (self.action_dim * self.N_horizon,), seed
        ).reshape(self.opt_state_shape) / normalising_factor


class DiffusionTrajOptProjection:
    def __init__(self, diff_optimiser: DiffusionOptimiser, env, horizon_length=300, mu=0.01):
        self.optimiser = diff_optimiser
        self.N_horizon = horizon_length
        self.action_dim = env.action_size
        self.opt_state_shape = (self.N_horizon, self.action_dim)
        self.env = env
        self.mu = mu

    def optimise_trajectory(self, init_state, seed, projection_fn, violation_higher_bound=None, alpha=1.0, normalising_factor=1.0, emerging_barrier=False):
        rollout = jax.jit(functools.partial(rollout_env, self.env))
        cost_func = jax.jit(
            lambda inputs, p: functools.partial(
                direct_transcription_cost,
                rollout,
                init_state,
            )(inputs.reshape(self.opt_state_shape)/normalising_factor)
        )
        return self.optimiser.reverse_process(
            cost_func, (self.action_dim * self.N_horizon,), seed, projection=projection_fn
        ).reshape(self.opt_state_shape) / normalising_factor
