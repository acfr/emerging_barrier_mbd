from collections import namedtuple
from random import randint
import functools

import jax
import jax.numpy as jnp
from jax import random

from jax_tqdm import loop_tqdm
from tqdm import trange

import matplotlib.pyplot as plt

from diffusion_trajopt.diffusion_utils import (
    alphas_from_betas,
    beta_schedule,
    get_alpha_bar_i,
)


DiffusionState = namedtuple(
    "DiffusionState", ["i", "Y_i", "rng_key", "mean_cost", "constraint_violations", "samples"])


class DiffusionOptimiser:
    def __init__(
        self,
        temperature=0.01,
        sample_size=2048,
        optimisation_steps=100,
        store_history=False,
        betas=None,
        noise=False,
    ):
        if betas is None:
            betas = beta_schedule(1e-4, 1e-2, optimisation_steps)
        self.alphas = alphas_from_betas(betas)
        self.alpha_bars = get_alpha_bar_i(self.alphas)
        self.store_history = store_history
        self.state_history = []
        self.temperature = temperature
        self.sample_size = sample_size
        self.noise = noise

    def reverse_process(
        self,
        fun,
        state_shape,
        seed,
        projection=None, 
        return_full=False
    ):
        self.state_history = []

        num_steps = len(self.alphas)
        rng_key = random.key(seed)
        Y_1 = random.normal(rng_key, shape=state_shape)
        _, rng_key = random.split(rng_key)

        # breakpoint()
        state = DiffusionState(Y_i=Y_1, i=num_steps,
                               rng_key=rng_key, mean_cost=0,
                               constraint_violations=0, samples=jnp.zeros((self.sample_size,) + state_shape))
        jitted_reverse_step = jax.jit(
            functools.partial(
                self.reverse_step,
                fun,
                alphas=self.alphas,
                alpha_bars=self.alpha_bars,
                sample_size=self.sample_size,
                temperature=self.temperature,
                projection=projection,
                noise=self.noise
            )
        )

        if self.store_history:
            # If history is needed
            for _ in trange(num_steps, 0, -1):
                state = jitted_reverse_step(state)
                self.state_history.append(state)
            # plt.plot(state.constraint_violations)
            return state.Y_i


        @loop_tqdm(num_steps)
        def loop_fun(i, val):
            return jitted_reverse_step(val)

        state = jax.lax.fori_loop(0, num_steps, loop_fun, state)
        
        if return_full:
            return state

        return state.Y_i

    @staticmethod
    def reverse_step(
        fun,
        carry: DiffusionState,
        alphas,
        alpha_bars,
        sample_size,
        temperature,
        projection=None,
        noise=False,
    ):
        alpha_i = alphas[carry.i]
        alpha_bar_i = alpha_bars[carry.i]
        alpha_bar_i_minus_1 = alpha_bars[carry.i - 1]

        # Sample around the current state for approximation
        std_dev = jnp.sqrt(1 / alpha_bar_i_minus_1 - 1)
        mean = carry.Y_i / jnp.sqrt(alpha_bar_i_minus_1)
        curl_Y = mean + std_dev * random.normal(
            key=carry.rng_key, shape=(sample_size, len(carry.Y_i))
        )
        # curl_Y = jnp.vstack(([mean], curl_Y))
        # breakpoint()

        prog = (len(alphas) - carry.i) / len(alphas)
        Js = jax.vmap(fun, in_axes=(0, None))(curl_Y, prog)
        constraint_violations = jnp.count_nonzero(jnp.isinf(Js))
        Js = jnp.nan_to_num(Js, posinf=1e9)

        # std = Js.std()
        # std = jnp.where(std < 1e-4, 1.0, std)
        # new_Js = (Js - Js.mean()) / std
        weights = jax.nn.softmax(-Js / temperature)
        # Guard against Js being all infty
        weights = jax.lax.cond(
            jnp.all(jnp.isnan(weights)),
            lambda: jnp.ones_like(weights)/len(weights),
            lambda: weights,
        )
        # weights = jnp.nan_to_num(weights)
        Y_bar_0 = jnp.einsum("n,ni->i", weights, curl_Y)

        score_approx = (
            1 / (1 - alpha_bar_i) * (-carry.Y_i +
                                     jnp.sqrt(alpha_bar_i) * Y_bar_0)
        )
        # breakpoint()
        z = random.normal(key=carry.rng_key, shape=(len(carry.Y_i),))
        Y_i = (
            1 / jnp.sqrt(alpha_i) * (carry.Y_i +
                                     (1 - alpha_bar_i) * score_approx)
        )

        if noise:
            Y_i = Y_i + jnp.sqrt(1 - alpha_i) * z
        if projection is not None:
            Y_i = projection(Y_i, prog)

        _, rng_key = random.split(carry.rng_key)
        return DiffusionState(
            i=carry.i - 1,
            Y_i=Y_i,
            rng_key=rng_key,
            mean_cost=jnp.min(Js),
            constraint_violations=constraint_violations,
            samples=curl_Y,
        )

