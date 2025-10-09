import jax.numpy as jnp
import jax

from .constraints import log_barrier

from dataclasses import dataclass


def default_barrier_schedule(prog, bound, alpha):
    start = bound
    end = 0
    end_time = 1
    c = start + (end - start)*((prog/end_time)**alpha)
    return jnp.max(jnp.array([0., c]))


@dataclass
class EmergingBarrierParams:
    higher_bound: float
    alpha: float
    end_time: float
    mu: float


def emerging_barrier_cost(
    constraint_dist,
    params: EmergingBarrierParams,
    progression,
):
    offset = default_barrier_schedule(
        progression, params.higher_bound, params.alpha)
    return log_barrier(constraint_dist + offset, params.mu)
