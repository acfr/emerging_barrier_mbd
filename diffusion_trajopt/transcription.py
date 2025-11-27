import jax.numpy as jnp
import jax


class DirectTranscription:
    def __init__(self, N_horizon, action_dim):
        self.N_horizon = N_horizon
        self.action_dim = action_dim

    def opt_flat_shape(self):
        return (self.N_horizon * self.action_dim,)

    def transcribe(self, opt_flat_state):
        return opt_flat_state.reshape(self.N_horizon, self.action_dim)
