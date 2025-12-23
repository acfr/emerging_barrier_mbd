import jax.numpy as jnp
import jax
import interpax


class DirectTranscription:
    def __init__(self, N_horizon, action_dim):
        self.N_horizon = N_horizon
        self.action_dim = action_dim

    def opt_flat_shape(self):
        return (self.N_horizon * self.action_dim,)

    def transcribe(self, opt_flat_state):
        return opt_flat_state.reshape(self.N_horizon, self.action_dim)


class SplineTranscription:
    def __init__(self, N_horizon, action_dim, num_points):
        self.N_horizon = N_horizon
        self.action_dim = action_dim
        self.num_points = num_points
        self.knot_points = jnp.linspace(0, 1, num_points)

    def opt_flat_shape(self):
        return (self.num_points * self.action_dim,)

    def transcribe(self, opt_flat_state):
        desired_points = jnp.linspace(0, 1, self.N_horizon)
        inputs = opt_flat_state.reshape((self.num_points, self.action_dim))
        return interpax.interp1d(desired_points, self.knot_points, inputs)
