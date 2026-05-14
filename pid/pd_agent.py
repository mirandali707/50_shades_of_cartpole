from typing import NamedTuple

class PolePDState(NamedTuple):
    """
    PD controller to keep the pole upright
    """
    goal: float # target pole angle
    k_p: float # P gain
    k_d: float # D gain
    last_error: float

def init_agent(k_p=8, k_d=100):
    return PolePDState(goal=0., k_p=k_p, k_d=k_d, last_error=0.)

def get_action(state, observation, act_key):
    """
    simple PD controller based on the pole angle (which we want to be 0)
    """
    x, x_dot, theta, theta_dot = observation
    error = state.goal - theta
    d_error = error - state.last_error
    ctrl_output = state.k_p * error + state.k_d * d_error
    action = 1 if ctrl_output < 0 else 0
    state = state._replace(last_error=error) # update state
    return state, action

# we already updated last_error in the get_action step, so no updating needed
# we're not looking at the reward or anything lol
def no_op(state, *args, **kwargs): return state

# state = reset_agent()
def reset_agent(_=None): return init_agent() # no learning, so reset is the same as init

# state = update(state, observation, reward, terminated, truncated, info)
update = no_op

model_name = "PD controller"