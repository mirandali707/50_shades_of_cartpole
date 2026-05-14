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
BATCH_SIZE = 8


class QNetwork(nnx.Module):
    def __init__(self, in_features, rngs: nnx.Rngs):
        # little baby MLP, 3-layers, 128 units per hidden layer
        # Q^(s, a)
        hidden_size = 128
        self.linear1 = nnx.Linear(in_features, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, 1, rngs=rngs)
    
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
    # input dim is observation size (4) + 1 for action
    # Q^(s, a; w)
    policy_network = QNetwork(OBS_SIZE + 1, nnx.Rngs(key))
    # Q^(s, a; w-)
    target_network = QNetwork(OBS_SIZE + 1, nnx.Rngs(key))

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
        obs_action = jnp.concatenate([
            jnp.array(observation, dtype=jnp.float32),
            jnp.array([action], dtype=jnp.float32),
        ])
        q_val = state.policy_network(obs_action)
        action_qvals.append(q_val[0])
    return int(jnp.argmax(jnp.array(action_qvals)))


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
    a_t  = batch[:, OBS_SIZE]
    r_t  = batch[:, OBS_SIZE + 1]
    s_t1 = batch[:, OBS_SIZE + 2: OBS_SIZE + 2 + OBS_SIZE]
    done = batch[:, -1]

    # compute targets: if terminal, target = r; else r + gamma * max_a Q_target(s', a)
    targets = []
    for i in range(batch_size):
        next_qvals = jnp.array([
            state.target_network(jnp.concatenate([s_t1[i], jnp.array([a])]))[0]
            for a in range(N_ACTIONS)
        ])
        bootstrap = r_t[i] + state.gamma * jnp.max(next_qvals)
        targets.append(jnp.where(done[i], r_t[i], bootstrap))
    targets = jnp.array(targets)  # (batch_size,)

    # gradient descent step on (target - policy_network preds) ** 2
    def loss_fn(policy_network):
        q_preds = jnp.array([
            policy_network(jnp.concatenate([s_t[i], a_t[i:i+1]]))[0]
            for i in range(batch_size)
        ])
        return jnp.mean((targets - q_preds) ** 2)

    grads = nnx.grad(loss_fn)(state.policy_network)

    # weight update: w = w - alpha * grad
    params = nnx.state(state.policy_network, nnx.Param)
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - state.alpha * g, params, grads
    )
    nnx.update(state.policy_network, new_params)

    # periodically sync target network <- policy network
    new_next_update = state.next_target_network_update - 1
    if new_next_update <= 0:
        nnx.update(state.target_network, nnx.state(state.policy_network))
        new_next_update = UPDATE_TARGET_NETWORK_EVERY
    state = state._replace(next_target_network_update=new_next_update)

    return state

