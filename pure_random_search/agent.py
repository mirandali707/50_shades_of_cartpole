from typing import NamedTuple
import jax.numpy as jnp
import numpy as np

model_name = "Pure Random Search"

SAVE_PATH = "pure_random_search_best.npz"

OBS_SIZE = 4
W_BOUNDS = (-1, 1)

class PureRandomSearchState(NamedTuple):
    """
    randomly sample w and keep track of the best w we've seen so far
    """
    best_w: jnp.ndarray
    best_reward: int
    curr_w: jnp.ndarray
    curr_reward: int


def init_agent():
    """
    returns initial state
    best_w and best_reward will be overridden after the first episode
    curr_w should be random
    """
    return PureRandomSearchState(
        best_w = jnp.zeros(OBS_SIZE),
        best_reward = -1,
        curr_w = jnp.array(np.random.uniform(W_BOUNDS[0], W_BOUNDS[1], OBS_SIZE)),
        curr_reward = 0
        )


def reset_agent(state):
    """
    between trials (episodes), check if best reward > curr_reward and if so, update best_w and best_reward
    """
    improved = state.curr_reward > state.best_reward
    return state._replace(
        best_w=state.curr_w if improved else state.best_w,
        best_reward=state.curr_reward if improved else state.best_reward,
        curr_w=jnp.array(np.random.uniform(W_BOUNDS[0], W_BOUNDS[1], OBS_SIZE)),
        curr_reward=0,
    )


def get_action(state, observation, act_key):
    """
    returns action based on current weights
    """
    return 0 if state.curr_w @ observation < 0 else 1


def update(state, 
           prev_observation, 
           action, 
           observation, 
           reward, 
           terminated, 
           truncated, 
           info):
    """
    updates model and returns new model state
    """
    state = state._replace(curr_reward=state.curr_reward + reward) # keep track of total reward for this episode
    return state

def save_trained_agent(state):
    """
    we just need to save our best weights, we'll use only those for eval
    """
    np.savez(SAVE_PATH, best_w=np.array(state.best_w))


def init_trained_agent(key=None):
    data = np.load(SAVE_PATH)
    best_w = jnp.array(data["best_w"])
    return PureRandomSearchState(
        best_w=best_w,
        best_reward=-1,
        curr_w=best_w,
        curr_reward=0,
    )

