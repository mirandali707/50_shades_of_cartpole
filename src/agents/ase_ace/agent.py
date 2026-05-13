from typing import NamedTuple
from src.encoders import boxes
from src.agents.base import AgentFns
from src.agents.ase_ace.ase import (
    ASEState, ase_init, ase_reset_traces, ase_get_action, ase_update,
)
from src.agents.ase_ace.ace import (
    ACEState, ace_init, ace_reset_traces, ace_get_r_hat, ace_update,
)

# --- combined state pytree ---
class ACEASEState(NamedTuple):
    ase: ASEState
    ace: ACEState

# --- pure functions matching the AgentFns interface ---
def init(cfg):
    return ACEASEState(
        ase=ase_init(boxes.N_BOXES),
        ace=ace_init(boxes.N_BOXES),
    )

def reset_episode(state):
    return ACEASEState(
        ase=ase_reset_traces(state.ase),
        ace=ace_reset_traces(state.ace),
    )

def act(state, obs, key, cfg):
    box_vec = boxes.encode(obs)
    action_pm = ase_get_action(state.ase, box_vec, key, cfg.sigma)
    action_env = ((action_pm + 1) // 2).astype(int)
    # Stash whatever update() needs:
    return action_env, {"box_vec": box_vec, "action_pm": action_pm}

def update(state, transition, cfg):
    box_vec      = transition.act_info["box_vec"]
    action_pm    = transition.act_info["action_pm"]
    box_vec_next = boxes.encode(transition.next_obs)

    # ACE
    r_hat, p = ace_get_r_hat(state.ace, box_vec_next, transition.reward, cfg.gamma)
    new_ace  = ace_update(state.ace, box_vec_next, r_hat, p, cfg.beta, cfg.lam)
    # ASE
    new_ase  = ase_update(state.ase, transition.act_info["box_vec"], action_pm, r_hat,
                          cfg.alpha, cfg.delta)
    return ACEASEState(ase=new_ase, ace=new_ace)

# --- the bundle the training loop consumes ---
def make_agent_fns(cfg):
    """Closes over cfg so the training loop sees a uniform 4-arg interface."""
    return AgentFns(
        init          = lambda: init(cfg),
        reset_episode = reset_episode,
        act           = lambda state, obs, key: act(state, obs, key, cfg),
        update        = lambda state, transition: update(state, transition, cfg),
    )