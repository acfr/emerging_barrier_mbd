import jax.scipy.optimize
import jax.numpy as jnp
import functools
import scipy
import jax
import numpy as np


def minimize_wrapper(actions_np, prog, rollout_fn):
    """
    Use scipy.minimize to project actions to 2d obstacle manifold. Only designed for ObstacleNavigator.

    Solve the following optimization problem:
    min_z ||z - u||^2
    s.t. rollout(z) does not intersect obstacles

    This wrapper is necessary because:
    - scipy.minimise can't be jitted through
    - jax.scipy.minimize is deprecated and cant do constraints (and only supports l-bfgs)
    - jaxopt.ScipyMinimize can't do general constraints, only simplified box constraints

    So we call scipy.minimise through an impure JAX io_callback
    """
    horizon, action_dim = actions_np.shape
    x0 = actions_np.flatten()

    def cost(z):
        """Objective function: squared L2 distance from original actions"""
        z_reshaped = z.reshape(horizon, action_dim)
        return np.sum((z_reshaped - actions_np) ** 2)

    def obstacle_constraint(z):
        """
        Constraint function: ensure no collision with obstacles
        Returns negative values when there are collisions (for inequality constraint)
        """
        z_reshaped = z.reshape(horizon, action_dim)

        # Convert to JAX arrays for rollout
        z_jax = jnp.array(z_reshaped)

        states, _, _ = rollout_fn(z_jax)

        # Check collision with each obstacle
        min_distances = states.contact.dist

        # Return the minimum distance (negative for constraint violation)
        # We want this to be >= 0 (no collision)
        return (min_distances).flatten() + (prog)*0.8

        # Set up constraints
    constraints = []
    constraints.append({
        'type': 'ineq',
        'fun': lambda z: obstacle_constraint(z),
        # 'jac': lambda z: jax.jacfwd(lambda z: obstacle_constraint(z))(z),
        'hessp': lambda z, dx: jax.hessian(lambda z: obstacle_constraint(z))(z).dot(dx),
    })

    # Run optimization
    return scipy.optimize.minimize(
        cost,
        x0,
        constraints=constraints,
        method='SLSQP', tol=1e-8,
        options={'disp': False, 'maxiter': 50, 'ftol': 1e-9}
    ).x.reshape(actions_np.shape)


def project_to_obstacle_avoiding_space(rollout_fn, actions, prog=0):
    """
    Args:
        rollout_fn: Function that takes (env, state_init, actions) and returns (states, stage_costs, terminal_cost)
        actions: (horizon, action_dim) - original actions to project

    Returns:
        projected_actions: (horizon, action_dim)
    """

    # Convert actions to numpy for scipy optimization
    actions_np = actions
    horizon, action_dim = actions_np.shape

    projected_actions = jax.experimental.io_callback(functools.partial(
        minimize_wrapper, rollout_fn=rollout_fn), actions_np, actions_np, prog)

    return projected_actions
