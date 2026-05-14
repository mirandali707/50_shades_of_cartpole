from typing import NamedTuple

class PolePState(NamedTuple):
    """
    P (proportional) controller to keep the pole upright
    """
    goal: float # target pole angle
    k_p: float # P gain

def init_agent(k_p=8):
    return PolePState(goal=0., k_p=k_p)

def get_action(state, observation, act_key):
    """
    simple PD controller based on the pole angle (which we want to be 0)
    """
    x, x_dot, theta, theta_dot = observation
    error = state.goal - theta
    ctrl_output = state.k_p * error 
    action = 1 if ctrl_output < 0 else 0
    return action
    
reset_agent =  init_agent # no learning, so reset is the same as init

# state = update(state, observation, reward, terminated, truncated, info)
def no_op(state, *args, **kwargs): return state
update = no_op

model_name = "P (proportional) controller"