from typing import NamedTuple
from flax import nnx
import jax.numpy as jnp
import jax
import numpy as np

N_ACTIONS = 2
OBS_SIZE = 4 # observation is x, x_dot, theta, theta_dot

EPSILON = 0.1
ALPHA = 0.001
GAMMA = 0.99
UPDATE_TARGET_NETWORK_EVERY = 200 # make larger if training for more iters
D_SIZE = 500
BATCH_SIZE = 16


class QNetwork(nnx.Module):
    def __init__(self, in_features, rngs: nnx.Rngs):
        # little baby MLP, 3-layers, 128 units per hidden layer
        # Q^(s) -> [Q(s,0), Q(s,1)]
        hidden_size = 128
        self.linear1 = nnx.Linear(in_features, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, N_ACTIONS, rngs=rngs)

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


def make_replay_tuple(prev_observation, action, reward, observation, terminated):
    """
    makes transition (s_t, a_t, r_t, s_t+1, done) for storage in replay buffer D
    """
    return jnp.concatenate([
        jnp.array(prev_observation, dtype=jnp.float32).flatten(),
        jnp.array([action], dtype=jnp.float32),
        jnp.array([reward], dtype=jnp.float32),
        jnp.array(observation, dtype=jnp.float32).flatten(),
        jnp.array([float(terminated)], dtype=jnp.float32),
    ])


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
    policy_network = QNetwork(OBS_SIZE, nnx.Rngs(key))
    target_network = QNetwork(OBS_SIZE, nnx.Rngs(key))

    # (s_t, a_t, r_t, s_t+1, done)
    # observation is 4d, action/reward/done are all scalars
    replay_tuple_size = OBS_SIZE + 1 + 1 + OBS_SIZE + 1
    D_replay_buffer = jnp.zeros((D_size, replay_tuple_size))
    return DQNState(
        eps=eps,
        alpha=alpha,
        gamma=gamma,
        policy_network=policy_network,
        target_network=target_network,
        next_target_network_update=next_target_network_update,
        D_size=D_size,
        D_replay_buffer=D_replay_buffer,
        next_replay_idx=0, # idx at which to store next replay tuple
        D_current_size=0 # num elems in replay buffer
    )


def reset_agent(state):
    """
    pass thru
    """
    return state


def get_action(state, observation, act_key):
    """
    returns action
    """
    explore_key, choice_key = jax.random.split(act_key)
    if jax.random.uniform(explore_key) < state.eps:
        # explore! fancy way of saying "pick 0 or 1 at random"
        return int(jax.random.randint(choice_key, (), 0, N_ACTIONS))
    # exploit! one forward pass gives Q-values for all actions
    obs = jnp.array(observation, dtype=jnp.float32)
    return int(jnp.argmax(state.policy_network(obs)))


@nnx.jit
def _train_step(policy_network, s_t, a_t, targets, alpha):
    def loss_fn(model):
        q_preds = model(s_t)  # (batch_size, N_ACTIONS)
        q_preds_taken = q_preds[jnp.arange(s_t.shape[0]), a_t]
        return jnp.mean((targets - q_preds_taken) ** 2)

    grads = nnx.grad(loss_fn)(policy_network)
    params = nnx.state(policy_network, nnx.Param)
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - alpha * g, params, grads
    )
    nnx.update(policy_network, new_params)


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
    # add current tuple to replay buffer (circular)
    replay_tuple = make_replay_tuple(prev_observation, action, reward, observation, terminated)
    idx = state.next_replay_idx % state.D_size
    new_D = state.D_replay_buffer.at[idx].set(replay_tuple)
    new_size = min(state.D_current_size + 1, state.D_size)
    state = state._replace(
        D_replay_buffer=new_D,
        next_replay_idx=(idx + 1) % state.D_size,
        D_current_size=new_size,
    )

    # need at least one sample before we can learn
    if state.D_current_size < 1:
        return state

    # sample minibatch from D
    batch_size = min(state.D_current_size, BATCH_SIZE)
    sample_key = jax.random.PRNGKey(state.next_replay_idx)
    indices = jax.random.randint(sample_key, (batch_size,), 0, state.D_current_size)
    batch = state.D_replay_buffer[indices]  # (batch_size, replay_tuple_size)

    # parse batch columns: s_t | a_t | r_t | s_t+1 | done
    s_t  = batch[:, :OBS_SIZE]
    a_t  = batch[:, OBS_SIZE].astype(jnp.int32)
    r_t  = batch[:, OBS_SIZE + 1]
    s_t1 = batch[:, OBS_SIZE + 2: OBS_SIZE + 2 + OBS_SIZE]
    done = batch[:, -1]

    # vectorized target computation: one forward pass for the whole batch
    next_qvals = state.target_network(s_t1)  # (batch_size, N_ACTIONS)
    max_next_q = jnp.max(next_qvals, axis=-1)  # (batch_size,)
    targets = jnp.where(done, r_t, r_t + state.gamma * max_next_q)

    # JIT-compiled gradient step (mutates policy_network in place)
    _train_step(state.policy_network, s_t, a_t, targets, state.alpha)

    # periodically sync target network <- policy network
    new_next_update = state.next_target_network_update - 1
    if new_next_update <= 0:
        nnx.update(state.target_network, nnx.state(state.policy_network))
        new_next_update = UPDATE_TARGET_NETWORK_EVERY
    state = state._replace(next_target_network_update=new_next_update)

    return state


def save_trained_agent(state):
    policy_state = nnx.state(state.policy_network)
    leaves = jax.tree_util.tree_leaves(policy_state)
    arrays = {f"w{i:04d}": np.array(leaf) for i, leaf in enumerate(leaves)}
    np.savez(f"{model_name}_best.npz", **arrays)


def init_trained_agent(key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    state = init_agent(key)
    saved = np.load(f"{model_name}_best.npz")
    saved_leaves = [jnp.array(saved[f"w{i:04d}"]) for i in range(len(saved.files))]

    _, policy_state = nnx.split(state.policy_network)
    treedef = jax.tree_util.tree_structure(policy_state)
    loaded_state = jax.tree_util.tree_unflatten(treedef, saved_leaves)

    nnx.update(state.policy_network, loaded_state)
    nnx.update(state.target_network, loaded_state)
    return state

