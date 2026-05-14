from typing import NamedTuple
import jax
import jax.numpy as jnp

from boxes import get_box, N_BOXES

N_ACTIONS = 2
EPSILON = 0.1
ALPHA = 0.3
GAMMA = 0.99

class QLearnerState(NamedTuple):
    q_table: jnp.ndarray  # (n_boxes, n_actions)
    eps: float
    alpha: float
    gamma: float

def init_q_learner(n_states=N_BOXES, n_actions=N_ACTIONS, eps=EPSILON, alpha=ALPHA, gamma=GAMMA):
    return QLearnerState(
        q_table=jnp.ones((n_states, n_actions)) * 100, # NOTE: initializing the q table with large values drives early exploration!
        eps=eps, alpha=alpha, gamma=gamma,
    )

def get_e_greedy_action(state, observation, act_key):
    explore_key, choice_key = jax.random.split(act_key)
    if jax.random.uniform(explore_key) < state.eps:
        # explore! fancy way of saying "pick 0 or 1 at random"
        return int(jax.random.randint(choice_key, (), 0, N_ACTIONS))
    # exploit! take the action that, based on this observation, we currently think will give us Best Future Returns
    box = get_box(observation)
    return int(jnp.argmax(state.q_table[box]))

def update_q_table(state, prev_observation, action, observation, reward, terminated, truncated, info):
    """
    update according to line 6 of the pseudocode from the image above;
    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a)) where s is the previous state and a is the action we just took
    except if the episode ends, our target is just the reward instead of expected discounted reward under optimal action
    """
    prev_box = get_box(prev_observation)
    new_box = get_box(observation)
    done = terminated or truncated
    target = reward if done else reward + state.gamma * jnp.max(state.q_table[new_box])
    update = state.alpha * (target - state.q_table[prev_box, action])
    new_q_table = state.q_table.at[prev_box, action].add(update)
    return state._replace(q_table=new_q_table)


no_op = lambda *args, **kwargs: None
def pass_thru(state, *args, **kwargs): return state

# state = init_agent()
init_agent = init_q_learner

# state = reset_agent(state)
reset_agent = pass_thru

# action = get_action(state, observation, act_key)
get_action = get_e_greedy_action

# state = update(state, observation, reward, terminated, truncated, info)
update = update_q_table

model_name = "Boxes Q-learning"