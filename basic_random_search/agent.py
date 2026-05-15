from typing import NamedTuple
import jax.numpy as jnp
import numpy as np

model_name = "Basic Random Search"

SAVE_PATH = "basic_random_search_best.npz"

OBS_SIZE = 4

ALPHA = 0.02 # step size
N_DIRECTIONS = 8 # number of directions sampled per iteration
NU = 0.03 # exploration noise std


class BasicRandomSearchState(NamedTuple):
    """
    maintain a current theta and improve it via finite-difference updates:
    each real "iteration" requires 2N trials / episodes: 
    for each of N directions delta_k,
    we run one rollout at theta + nu*delta_k and one at theta - nu*delta_k
    (symmetrically perturb theta by delta_k in either direction +/-)
    after 2N episodes, update theta and resample directions.
    """
    theta: jnp.ndarray # current parameter vector
    deltas: jnp.ndarray # (N, OBS_SIZE) sampled directions for this iteration
    rewards_plus: jnp.ndarray # (N,) rewards for + perturbations
    rewards_minus: jnp.ndarray # (N,) rewards for - perturbations
    rollout_idx: int # which of the 2N rollouts we're on (0 .. 2N-1)
    curr_reward: float # reward accumulated in the current episode
    curr_w: jnp.ndarray # the perturbed weights actually used for this episode


def _sample_directions():
    return jnp.array(np.random.randn(N_DIRECTIONS, OBS_SIZE))


def _current_w(state):
    """
    get policy weights to use for the current rollout.
    rollouts are ordered as: (k=0,+), (k=0,-), (k=1,+), (k=1,-), ...
    so k = rollout_idx // 2 and sign = + if rollout_idx % 2 == 0 else -.
    """
    k = state.rollout_idx // 2
    sign = 1.0 if state.rollout_idx % 2 == 0 else -1.0
    return state.theta + sign * NU * state.deltas[k]


def init_agent():
    """
    theta_0 = 0 per the BRS algorithm
    sample the first batch of N directions
    so the first episode can roll out with theta + nu*delta_0
    """
    theta = jnp.zeros(OBS_SIZE)
    deltas = _sample_directions()
    # compute the w for rollout_idx=0 up front so the train loop can read it
    # before the first episode runs
    curr_w = theta + NU * deltas[0]
    return BasicRandomSearchState(
        theta=theta,
        deltas=deltas,
        rewards_plus=jnp.zeros(N_DIRECTIONS),
        rewards_minus=jnp.zeros(N_DIRECTIONS),
        rollout_idx=0,
        curr_reward=0.0,
        curr_w=curr_w,
    )


def reset_agent(state):
    """
    called between episodes. 
    here there are 2 cases:
    1. mid-iteration (still have rollouts to do in this batch of 2N):
       record the just-finished episode's reward into rewards_plus/rewards_minus,
       advance rollout_idx, zero curr_reward.
    
    2. done with this iteration (that was the last (2N-th) rollout):
       record the reward, do the BRS theta update, resample directions, reset.
    
    in either case, refresh curr_w to whatever the next episode will roll out.
    """
    k = state.rollout_idx // 2
    is_plus = state.rollout_idx % 2 == 0
    
    # Record this episode's reward into the appropriate slot.
    rewards_plus = state.rewards_plus
    rewards_minus = state.rewards_minus
    if is_plus:
        rewards_plus = rewards_plus.at[k].set(state.curr_reward)
    else:
        rewards_minus = rewards_minus.at[k].set(state.curr_reward)
    
    next_idx = state.rollout_idx + 1
    
    if next_idx < 2 * N_DIRECTIONS:
        # mid-iteration, update list of rewards with this rollout and continue
        new_state = state._replace(
            rewards_plus=rewards_plus,
            rewards_minus=rewards_minus,
            rollout_idx=next_idx,
            curr_reward=0.0,
        )
        return new_state._replace(curr_w=_current_w(new_state))
    
    # done with this iteration: apply the BRS update
    #   theta_{j+1} = theta_j + (alpha / N) * sum_k [r(+) - r(-)] * delta_k
    reward_diffs = rewards_plus - rewards_minus # (N,)
    update = (ALPHA / N_DIRECTIONS) * (reward_diffs @ state.deltas) # matrix-vector pdt: (N,) @ (N, D) = (D,)
    new_theta = state.theta + update
    new_deltas = _sample_directions()
    
    new_state = BasicRandomSearchState(
        theta=new_theta,
        deltas=new_deltas,
        rewards_plus=jnp.zeros(N_DIRECTIONS),
        rewards_minus=jnp.zeros(N_DIRECTIONS),
        rollout_idx=0,
        curr_reward=0.0,
        curr_w=jnp.zeros(OBS_SIZE),  # placeholder, overwritten on next line
    )
    return new_state._replace(curr_w=_current_w(new_state))


def get_action(state, observation, act_key):
    """
    use perturbed weights for the current rollout
    """
    w = state.curr_w
    return 0 if w @ observation < 0 else 1


def update(state,
           prev_observation,
           action,
           observation,
           reward,
           terminated,
           truncated,
           info):
    """
    accumulate reward for the current episode, 
    since actual theta update happens in reset_agent 
    at the end of each 2N-episode iteration.
    """
    return state._replace(curr_reward=state.curr_reward + reward)


def save_trained_agent(best_w):
    """
    train loop tracks the best w across all episodes and passes it in directly.
    """
    np.savez(SAVE_PATH, best_w=np.array(best_w))


def init_trained_agent(key=None):
    data = np.load(SAVE_PATH)
    best_w = jnp.array(data["best_w"])
    # at eval time curr_w IS the trained weight vector; theta/deltas are unused
    return BasicRandomSearchState(
        theta=best_w,
        deltas=jnp.zeros((N_DIRECTIONS, OBS_SIZE)),
        rewards_plus=jnp.zeros(N_DIRECTIONS),
        rewards_minus=jnp.zeros(N_DIRECTIONS),
        rollout_idx=0,
        curr_reward=0.0,
        curr_w=best_w,
    )