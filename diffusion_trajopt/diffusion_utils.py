import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def alphas_from_betas(betas):
    return 1 - betas


@partial(jax.jit, static_argnums=2)
def beta_schedule(beta_1: float, beta_T: float, T: int):
    return jnp.linspace(beta_1, beta_T, T)


@jax.jit
def get_alpha_bar_i(alpha_sched):
    return jnp.cumprod(alpha_sched)
