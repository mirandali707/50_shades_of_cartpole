from typing import NamedTuple


model_name = "DQN"


class DQNState(NamedTuple):
    """
    model state for DQN
    """
    foo : str


def init_agent():
    """
    returns initial state
    """
    return DQNState(foo="foo")


def reset_agent(state):
    """
    returns resetted state
    if pass through, don't change this fn
    """
    return state


def get_action(state, observation, act_key):
    """
    returns action
    """
    return 1


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
    state = state._replace(foo="bar") # update state
    return state

