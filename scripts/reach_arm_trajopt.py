import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from random import randint
from diffusion_trajopt.diffusion_opt import DiffusionOptimiser
from diffusion_trajopt.trajopt import DiffusionTrajOpt, rollout_env
from reach_mjx.alpha_bluerov_brax_env import ReachArm
from reach_mjx.viser_viz import ReachEnvViz
from jax import config

import argparse
import pickle
import time

# config.update("jax_enable_x64", True)
# config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# config.update("jax_persistent_cache_min_compile_time_secs", 0)
# config.update("jax_persistent_cache_enable_xla_caches", "all")
#


def get_wrist_val(traj):
    return traj.xpos[:, sys.wrist_id, :]


def get_wrists_val(traj):
    return traj.xpos[:, :, sys.wrist_id, :]


def add_optimiser_trajectories(
    trajopt_object, sys, state_init, viz_object, normalising_factor
):
    # time.sleep(20)
    for i, val in enumerate(trajopt_object.optimiser.state_history):
        actions = (
            jax.vmap(trajopt_object.transcription.transcribe)(val.samples)
            / normalising_factor
        )
        traj = jax.vmap(rollout_env, in_axes=(None, None, 0))(
            sys, state_init, actions[::3, :, :]
        )[0]
        points = get_wrists_val(traj)

        if i == 0:
            lineseg_handle = viz_object.ui.add_trajectories(
                points,
                f"traj_{i}",
                colors=[100, 0, 0],
                line_width=2,
            )
        else:
            viz_object.ui.modify_trajectories(lineseg_handle, points)
        # time.sleep(0.1)
        print(f"Adding {i}-th traj")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emerging_barrier", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=1.3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--normalising_factor", type=float, default=45.0)
    parser.add_argument("--horizon", type=int, default=100)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    rng = jax.random.key(100)

    sys = ReachArm(rng, target_pos=jnp.array([0.6, 0.0, -0.1]))
    opt = DiffusionOptimiser(
        temperature=args.temperature,
        optimisation_steps=args.steps,
        sample_size=args.samples,
        store_history=True,
    )

    step = jax.jit(lambda s, a: sys.step(s, a))
    reset_env = jax.jit(lambda r: sys.reset(r))

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)
    state_init = step(state_init, jnp.zeros((sys.action_size,)))
    trajopt = DiffusionTrajOpt(opt, sys, args.horizon, mu=args.mu)

    normalising_factor = args.normalising_factor

    if args.emerging_barrier:
        actions_final = trajopt.optimise_trajectory(
            state_init,
            args.seed,
            normalising_factor=normalising_factor,
            emerging_barrier=args.emerging_barrier,
            violation_higher_bound=0.15,
            alpha=args.alpha,
        )
    else:
        actions_final = trajopt.optimise_trajectory(
            state_init, args.seed, normalising_factor=normalising_factor
        )
        actions_final = trajopt.optimise_trajectory(
            state_init, args.seed, normalising_factor=normalising_factor
        )

    # Use MJX trajectory playback mode
    traj, stage_costs, terminal = rollout_env(sys, state_init, actions_final)
    meets_constraints = not sys.is_colliding(traj).any()
    print(
        "TRAJ COST: {}\nRESULT: {}\nTERM_DIST: {}".format(
            (jnp.sum(stage_costs) + terminal),
            ("PASSED" if meets_constraints else "FAILED"),
            jnp.linalg.norm(get_wrist_val(traj)[-1, :] - sys.target_pos),
        )
    )

    T = traj.qpos.shape[0]
    mjx_traj = [jax.tree.map(lambda x: x[i], traj) for i in range(T)]

    if args.viz:
        plt.plot([x.constraint_violations for x in trajopt.optimiser.state_history])
        plt.show()

        viz = ReachEnvViz(hz=20, target_pos=sys.target_pos)
        viz.reinit_sim(state_init, actions_final)
        add_optimiser_trajectories(trajopt, sys, state_init, viz, normalising_factor)
        # viz.paused = False
        viz.run_sim()
