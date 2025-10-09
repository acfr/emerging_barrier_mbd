import jax

# evaluate the diffused uss
def eval_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews


def rollout_us_arr(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipeline_states) = jax.lax.scan(step, state, us)
    # add original state
    pipeline_states = jax.numpy.concatenate(
        [jax.numpy.expand_dims(state.pipeline_state, axis=0), pipeline_states], axis=0
    )
    return rews, pipeline_states


def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states
