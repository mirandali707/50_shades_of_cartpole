from typing import NamedTuple
from flax import nnx
import jax.numpy as jnp
import jax

N_ACTIONS = 2
OBS_SIZE = 4 # observation is x, x_dot, theta, theta_dot

EPSILON = 0.1
ALPHA = 0.3
GAMMA = 0.99
UPDATE_TARGET_NETWORK_EVERY = 100 # make larger if training for more iters
D_SIZE = 20


class QNetwork(nnx.Module):
    def __init__(self, in_features, n_actions, rngs: nnx.Rngs):
        # little baby MLP, 3-layers, 128 units per hidden layer
        hidden_size = 128
        self.linear1 = nnx.Linear(in_features, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, n_actions, rngs=rngs)
    
    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)  # no activation on output because Q values can be any value
        return x


model_name = "DQN"
class DQNState(NamedTuple):
    """
    model state for DQN
    """
    eps: float
    alpha: float
    gamma: float
    policy_network: QNetwork # use this to figure out what action to take, has weights w in pseudocode
    target_network: QNetwork # use this to compute the target, has weights w- in pseudocode
    next_target_network_update: int
    D_size: int 
    D_replay_buffer: jnp.ndarray
    next_replay_idx: int
    D_current_size: int


def make_replay_tuple(prev_observation, action, reward, observation):
    """
    makes transition (s_t, a_t, r_t, s_t+1) for storage in replay buffer D
    """
    pass
    # TODO return flattened replay tuple as jnp array


def init_agent(key, 
               eps=EPSILON, 
               alpha=ALPHA, 
               gamma=GAMMA,
               next_target_network_update=UPDATE_TARGET_NETWORK_EVERY,
               D_size=D_SIZE
               ):
    """
    returns initial state
    """
    # input dim is observation size (4) + 1 for action
    # Q^(s, a; w)
    policy_network = QNetwork(OBS_SIZE + 1, 2, nnx.Rngs(key))
    # Q^(s, a; w-)
    target_network = QNetwork(OBS_SIZE + 1, 2, nnx.Rngs(key))

    # (s_t, a_t, r_t, s_t+1)
    # observation is 4d, action and reward are both scalars
    replay_tuple_size = OBS_SIZE + 1 + 1 + OBS_SIZE
    D_replay_buffer=jnp.zeros((D_size, replay_tuple_size))
    return DQNState(
        eps=eps,
        alpha=alpha,
        gamma=gamma,
        policy_network=policy_network,
        target_network=target_network,
        next_target_network_update=next_target_network_update,
        D_size=D_replay_buffer,
        next_replay_idx=0, # idx at which to store next replay tuple
        D_current_size=0 # num elems in replay buffer
    )


def reset_agent(state):
    """
    reset replay buffer (set size and next idx to 0)
    do NOT reset policy, target networks - that's how learnign happens!
    """
    state = state._replace(
        next_replay_idx=0,
        D_current_size=0
        )
    return state


def get_action(state, observation, act_key):
    """
    returns action
    """
    explore_key, choice_key = jax.random.split(act_key)
    if jax.random.uniform(explore_key) < state.eps:
        # explore! fancy way of saying "pick 0 or 1 at random"
        return int(jax.random.randint(choice_key, (), 0, N_ACTIONS))
    # exploit! take the action that, based on this observation, we currently think will give us Best Future Returns
    action_qvals = []
    for action in range(N_ACTIONS):
        # TODO make obs, action pair
        # TODO pass through state.policy_network
        pass
    return int(jnp.argmax(action_qvals))


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
    # TODO!
    # add current tuple to replay buffer
    # sample minibatch from D
    # if episode terminated, then target = reward
    # otherwise target is reward + expected discounted reward (from target net)
    # gradient descent step on (target - policy_network preds) **2
    # weight update is alpha * grad
    return state

