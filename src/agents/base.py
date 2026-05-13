from typing import NamedTuple, Any, Callable

class AgentFns(NamedTuple):
    """
    functional interface for all models/methods, so we can reuse train/eval code!
    functional programming so we can use jax: all fns take state, return new state.
    """
    init:          Callable[..., Any]                       # config -> state
    reset_episode: Callable[[Any], Any]                     # state -> state
    act:           Callable[[Any, Any, Any], tuple]         # (state, obs, key) -> (action_env, act_info)
    update:        Callable[..., Any]                       # (state, transition) -> state


class Transition(NamedTuple):
    obs: Any
    action: int
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool
    act_info: Any        # whatever act() returned alongside the action