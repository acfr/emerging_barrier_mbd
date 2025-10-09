from diffusion_trajopt.diffusion_opt import DiffusionOptimiser
from diffusion_trajopt.emerging_barrier import EmergingBarrierParams, emerging_barrier_cost

import scienceplots
import jax.numpy as jnp
import jax

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use(['science', 'no-latex'])
if __name__ == "__main__":
    T = 5
    optimiser = DiffusionOptimiser(sample_size=100, temperature=T, noise=False)
    max_bound = 4.0
    params = EmergingBarrierParams(
        higher_bound=max_bound, alpha=1.4, mu=6, end_time=1)

    def objective_function(x, p):
        constraint_dist = jnp.linalg.norm(x) - max_bound
        return jnp.linalg.norm(x - 1)**2 + emerging_barrier_cost(constraint_dist, params, p)

    ntrajs = 400
    results = jax.vmap(optimiser.reverse_process, in_axes=(None, None, 0))(
        objective_function, (1,), jnp.arange(0, ntrajs))
    # breakpoint()
    y_map = jax.vmap(objective_function, in_axes=(0, None))
    # plt.plot(x, jnp.exp(-y/T)*20)
    # breakpoint()
    # plt.plot(x,y)
    # plt.show()
    # breakpoint()
    # counts, bins = jnp.histogram(results, bins=150)
    # plt.stairs(counts, bins)

    for j in range(0, ntrajs):
        traj = []
        for i in range(0, 100):
            traj.append(optimiser.state_history[i].Y_i.val[j])
        plt.plot(range(0, 100), traj, color='red', alpha=0.05)

    x = jnp.linspace(-10, 10, 400)
    img = np.zeros((400, 500))
    for i in range(0, 99):
        img[:, i] = y_map(x, i/500)

    plt.imshow(img, extent=[0, 100, min(x), max(x)], aspect=4, cmap='cividis_r')

    # plt.scatter(jnp.arange(0,100), results)
    plt.show()
