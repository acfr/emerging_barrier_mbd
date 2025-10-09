from typing import List, Union, Optional, Tuple, Dict
import numpy as np
import jax.numpy as jnp
import jax
import equinox as eqx
from typing import Union, List, Optional, Tuple, NamedTuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@jax.jit
def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


class ContactInfo(NamedTuple):
    """Contact information for MJX compatibility."""
    dist: jax.Array  # Distance to obstacles


class NavigatorState(NamedTuple):
    """State for the ObstacleNavigator environment."""
    position: jax.Array  # 2D position [x, y]
    contact: ContactInfo  # Contact information for MJX compatibility


class ObstacleNavigator(eqx.Module):
    """2D obstacle navigation environment following the same API as ReachArm."""

    max_step_size: float
    target_position: jax.Array
    obstacle_centers: jax.Array
    obstacle_radii: jax.Array

    def __init__(
        self,
        rng,
        step_size=0.5,
        target_position=None,
        obstacles=True,
        obstacle_config=None,
    ):
        self.max_step_size = step_size
        self.target_position = (
            jnp.array([5.0, 2.0])
            if target_position is None
            else jnp.array(target_position)
        )

        # Set up obstacles
        if obstacles:
            if obstacle_config is None:
                # Default obstacle grid configuration
                grid_x_min, grid_x_max = -4.0, 10.0
                grid_y_min, grid_y_max = -4.0, 4.0
                grid_spacing = 2.0
                obstacle_radius = 0.5
            else:
                # Unpack provided configuration
                grid_x_min = obstacle_config.get("x_min", -6.0)
                grid_x_max = obstacle_config.get("x_max", 10.0)
                grid_y_min = obstacle_config.get("y_min", -6.0)
                grid_y_max = obstacle_config.get("y_max", 6.0)
                grid_spacing = obstacle_config.get("spacing", 2.0)
                obstacle_radius = obstacle_config.get("radius", 0.5)

            # Generate obstacle positions
            obstacles = self._generate_obstacles(
                grid_x_min, grid_x_max, grid_y_min, grid_y_max,
                grid_spacing, obstacle_radius
            )
        else:
            obstacles = []

        # Pre-compute JAX arrays for obstacle data
        self.obstacle_centers = jnp.array(
            [[obs[0], obs[1]] for obs in obstacles])
        self.obstacle_radii = jnp.array([obs[2] for obs in obstacles])

    def _generate_obstacles(self, x_min, x_max, y_min, y_max, spacing, radius):
        """Generate a list of obstacles in the format (center_x, center_y, radius)."""
        x_positions = jnp.arange(x_min, x_max, spacing)
        y_positions = jnp.arange(y_min, y_max, spacing)

        obstacles = []
        for y in y_positions:
            for x in x_positions:
                obstacles.append((float(x), float(y), float(radius)))

        return obstacles

    def reset(self, rng: jax.Array) -> NavigatorState:
        """Reset the environment to initial state."""
        # Add some randomness to initial position
        base_pos = jnp.array([-1.0, -2.0])
        # noise = jax.random.uniform(rng, (2,), minval=-0.5, maxval=0.5)
        initial_position = base_pos

        # Create initial state with contact information
        initial_state = NavigatorState(
            position=initial_position, contact=ContactInfo(dist=jnp.array([0.0])))
        # Update contact with actual SDF value
        contact_dist = self.sdf_fn(initial_state)
        return NavigatorState(position=initial_position, contact=ContactInfo(dist=contact_dist))

    @property
    def action_size(self):
        return 2

    def step(self, state: NavigatorState, action: jax.Array) -> NavigatorState:
        """Run one timestep of the environment's dynamics."""
        # Handle the case where action magnitude is zero
        action_norm = jnp.linalg.norm(action)
        normalized_action = jnp.where(
            action_norm > 0, action / action_norm, jnp.zeros_like(action)
        )

        new_step = sigmoid(action_norm / 10) * \
            self.max_step_size * normalized_action
        new_position = state.position + new_step

        # Create new state with updated position
        new_state = NavigatorState(
            position=new_position, contact=ContactInfo(dist=jnp.array([0.0])))
        # Update contact with actual SDF value
        contact_dist = self.sdf_fn(new_state)
        return NavigatorState(position=new_position, contact=ContactInfo(dist=contact_dist))

    @staticmethod
    def is_colliding(state: NavigatorState) -> jax.Array:
        """Check if the agent at the given state collides with any obstacle."""
        # This method should return collision distances like mjx.Data.contact.dist
        # For now, we'll return a dummy array with the SDF value
        # In practice, this should be compatible with the collision checking in trajopt.py
        return jnp.array([0.0])  # Placeholder - will be overridden by sdf_fn

    def sdf_fn(self, state: NavigatorState) -> jax.Array:
        """Signed distance function to obstacles."""
        if len(self.obstacle_centers) == 0:
            return jnp.array([1000.0])  # No obstacles, large positive distance

        distances = jnp.linalg.norm(
            self.obstacle_centers - state.position, axis=1)
        signed_distances = distances - self.obstacle_radii
        # jax.debug.print("Distances {}", signed_distances)
        return jnp.array([jnp.min(signed_distances)])

    def stage_cost(self, state: NavigatorState, action: jax.Array) -> jax.Array:
        """Calculate stage cost for a single step."""
        # Base cost is distance to target
        position_cost = jnp.linalg.norm(state.position - self.target_position)
        action_cost = jnp.linalg.norm(action) / 10

        # base_cost = position_cost + action_cost
        #
        # # Add collision penalty
        # sdf_value = self.sdf_fn(state)
        # collision_penalty = jnp.where(sdf_value <= 0, 1000.0, 0.0)
        # total_cost = base_cost + collision_penalty
        #
        # # Convert boolean to JAX array for conditional logic
        # zero_cost_flag = jnp.array(self.zero_cost_col, dtype=jnp.bool_)
        # result = jnp.where(zero_cost_flag, 0.0, total_cost)
        return position_cost + action_cost

    def terminal_cost(self, state: NavigatorState) -> jax.Array:
        """Calculate terminal cost."""
        return self.stage_cost(state, jnp.array([0.0, 0.0]))*20

    def get_obstacles(self):
        """Return obstacles in the format expected by render_trajectory."""
        obstacles = []
        for i in range(len(self.obstacle_centers)):
            obstacles.append((
                (self.obstacle_centers[i, 0]).astype(float),
                (self.obstacle_centers[i, 1]).astype(float),
                (self.obstacle_radii[i]).astype(float)
            ))
        return obstacles


# Custom rollout function for ObstacleNavigator that's compatible with DiffusionTrajOptMJX
@jax.jit
def rollout_navigator(env, state_init, actions):
    """Rollout function for ObstacleNavigator that mimics rollout_env from trajopt.py."""
    def step(state, u):
        state = env.step(state, u)
        return state, (state, env.stage_cost(state, u))

    terminal_state, (states, stage_costs) = jax.lax.scan(
        step, state_init, actions[:-1])
    return states, stage_costs, env.terminal_cost(env.step(terminal_state, actions[-1]))


def render_trajectory(
    states: Union[List[NavigatorState], List[jnp.ndarray], jnp.ndarray],
    target: jnp.ndarray = jnp.array([5, 2]),
    obstacles: Optional[List[Tuple[float, float, float]]] = None,
    figsize: Tuple[int, int] = (10, 8),
    animate: bool = False,
    interval: float = 0.1,
    save_path: Optional[str] = None,
):
    """
    Render a 2D trajectory for the ObstacleNavigator environment.

    Args:
        states: List of NavigatorState objects or list of 2D positions
        target: Target position as a 2-element array [x, y]
        obstacles: List of obstacles, each defined as (center_x, center_y, radius)
        figsize: Figure size as (width, height)
        animate: If True, animate the trajectory (requires display in notebook)
        interval: Time interval between animation frames in seconds
        save_path: Path to save the figure, if provided

    Returns:
        matplotlib figure and axes
    """
    # Extract positions from states if they are NavigatorState objects
    positions = jnp.array([state for state in states.position])  # type: ignore

    # Convert inputs to numpy arrays for matplotlib
    target_np = jnp.array(target)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot target
    ax.scatter(
        target_np[0], target_np[1], color="green", s=100, marker="*", label="Target"
    )

    # Plot obstacles if provided
    if obstacles is not None:
        for obs in obstacles:
            circle = patches.Circle(
                (obs[0], obs[1]), obs[2], color="red", alpha=0.5)
            ax.add_patch(circle)

    # Determine axis limits with padding
    all_x = jnp.append(positions[:, 0], target_np[0])
    all_y = jnp.append(positions[:, 1], target_np[1])

    min_x, max_x = float(jnp.min(all_x) - 1), float(jnp.max(all_x) + 1)
    min_y, max_y = float(jnp.min(all_y) - 1), float(jnp.max(all_y) + 1)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Set labels and title
    # ax.set_xlabel("X Position")
    # ax.set_ylabel("Y Position")
    # ax.set_title("2D ObstacleNavigator Trajectory")
    # ax.grid(True)
    # ax.set_ticks()

    # Plot starting point
    ax.scatter(
        positions[0, 0], positions[0, 1], color="blue", s=100, marker="o", label="Start"
    )

    if animate and len(positions) > 1:
        from IPython.display import clear_output

        # Plot initial state
        (trajectory_line,) = ax.plot(
            positions[0:1, 0], positions[0:1, 1], "b-", alpha=0.7
        )
        current_point = ax.scatter(
            positions[0, 0], positions[0, 1], color="blue", s=50)

        plt.legend()
        plt.tight_layout()

        # Animate the trajectory
        for i in range(1, len(positions)):
            # Update trajectory line
            trajectory_line.set_data(
                positions[: i + 1, 0], positions[: i + 1, 1])

            # Update current position
            current_point.set_offsets(positions[i])

            plt.draw()
            plt.pause(interval)
    else:
        # Plot entire trajectory at once
        ax.plot(positions[:, 0], positions[:, 1],
                "b-", alpha=0.7, label="Trajectory")

        # Plot end point
        if len(positions) > 1:
            ax.scatter(
                positions[-1, 0],
                positions[-1, 1],
                color="red",
                s=100,
                marker="x",
                label="End",
            )

    plt.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    # square aspect
    ax.set_aspect("equal", "box")

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def render_multiple_trajectories(
    trajectories: Union[List[Union[List[NavigatorState], jnp.ndarray]], jnp.ndarray, NavigatorState],
    target: jnp.ndarray = jnp.array([5, 2]),
    obstacles: Optional[List[Tuple[float, float, float]]] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Union[str, Tuple[float, float, float, float]]]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    title: str = "Multiple Trajectories",
    show_start_end: bool = True,
    grid: bool = True,
    traj_alpha: float = 0.8,
    horizontal=False,
):
    """
    Render multiple 2D trajectories for the ObstacleNavigator environment.

    Args:
        trajectories: List of trajectories, batched array, or NavigatorState with batched positions
        target: Target position as a 2-element array [x, y]
        obstacles: List of obstacles, each defined as (center_x, center_y, radius)
        labels: Labels for each trajectory in the legend
        colors: Colors for each trajectory
        figsize: Figure size as (width, height)
        save_path: Path to save the figure, if provided
        title: Plot title
        show_start_end: Whether to highlight start and end points
        grid: Whether to show grid lines

    Returns:
        matplotlib figure and axes
    """
    # Handle different input types
    if isinstance(trajectories, NavigatorState):
        # NavigatorState with batched positions: shape (num_trajectories, num_states, 2)
        positions_array = trajectories.position
        if positions_array.ndim == 3:
            # Extract individual trajectories
            positions_list = [positions_array[i]
                              for i in range(positions_array.shape[0])]
        else:
            raise ValueError(
                f"Unexpected NavigatorState position shape: {positions_array.shape}")

    elif isinstance(trajectories, jnp.ndarray):
        # Direct batched array
        if trajectories.ndim == 3:
            # Shape: (num_trajectories, num_states, state_dim)
            positions_list = [trajectories[i]
                              for i in range(trajectories.shape[0])]
        else:
            raise ValueError(
                f"Unexpected trajectories array shape: {trajectories.shape}")

    else:
        # List of trajectories
        positions_list = []
        for traj in trajectories:
            if len(traj) > 0 and isinstance(traj[0], NavigatorState):
                # Extract positions from NavigatorState objects
                positions = jnp.array([state.position for state in traj])
            else:
                # Assume it's already position data
                positions = jnp.array(traj)
            positions_list.append(positions)

    # Set up default values
    num_trajectories = len(positions_list)

    # if labels is None:
    #     labels = [f"Trajectory {i+1}" for i in range(num_trajectories)]

    if colors is None:
        # Use a qualitative colormap
        colormap = plt.cm.get_cmap("tab10")
        colors = [colormap(i % 1) for i in range(num_trajectories)]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot obstacles if provided
    if obstacles is not None:
        for obs in obstacles:
            circle = patches.Circle(
                (obs[0], obs[1]), obs[2],
                color="red", alpha=0.6, zorder=5
            )
            ax.add_patch(circle)

    # Plot each trajectory
    for i, positions in enumerate(positions_list):
        # Plot trajectory line
        ax.plot(
            positions[:, 0], positions[:, 1],
            color=colors[i], alpha=traj_alpha, linewidth=2,
            zorder=i + 1, label=labels[i] if labels is not None else None
        )

        # Highlight start and end points if requested
        if show_start_end:
            # Start point (triangle)
            ax.scatter(
                positions[0, 0], positions[0, 1],
                color=colors[i], s=100, marker="^",
                zorder=i + 20
            )
            # # End point (X marker)
            # ax.scatter(
            #     positions[-1, 0], positions[-1, 1],
            #     color=colors[i], s=100, marker="x",
            #     zorder=i + 20
            # )

    # Set axis limits and properties
    # Plot target
    target_np = jnp.array(target)
    ax.scatter(
        target_np[0], target_np[1],
        color="blue", s=150, marker="*",
        label="Target", zorder=1000
    )

    all_positions = jnp.vstack(positions_list + [target_np.reshape(1, -1)])

    min_x, max_x = jnp.min(
        all_positions[:, 0]) - 1, jnp.max(all_positions[:, 0]) + 1
    min_y, max_y = jnp.min(
        all_positions[:, 1]) - 1, jnp.max(all_positions[:, 1]) + 1

    # Use fixed limits for consistency
    min_x, max_x = -4, 6
    min_y, max_y = -3, 4

    if horizontal:
        min_x, max_x = -2, 3.5
        min_y, max_y = -2.5, 5.5

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', 'box')

    # Remove axis ticks for cleaner look
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])

    # Add grid if requested
    if grid:
        ax.grid(True)

    # Add legend and title
    ax.legend(loc="upper left", framealpha=0.7)
    # ax.set_title(title, fontsize=14)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
