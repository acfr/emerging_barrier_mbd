import jax
import jax.numpy as jnp

def extended_log(x):
    return jax.lax.cond(x <= 0, lambda: -jnp.inf, lambda: jnp.log(x))
    return jax.lax.cond(x <= 0, lambda: -jnp.inf, lambda: jnp.log(x))
    return jax.lax.select(x <= 0, -1000.0*jnp.ones_like(x), jnp.log(x))

def log_barrier(val, mu):
    return -mu * extended_log(val)
