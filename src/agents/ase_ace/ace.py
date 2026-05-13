from typing import NamedTuple
import jax.numpy as jnp

class ACEState(NamedTuple):
    """
    Adaptive Critic Element;
    helps with credit assignment by computing internal reward from box vec + reinforcement signal (-1 if failure, 0 otherwise)
    """
    v: jnp.ndarray # (n_boxes,)
    x_bar: jnp.ndarray # (n_boxes,)
    p_prev: jnp.ndarray # scalar — p(t-1), previous prediction

def ace_init(n_boxes):
    """
    initialize an all-zero starting state for ACE 
    """
    return ACEState(
        v=jnp.zeros(n_boxes),
        x_bar=jnp.zeros(n_boxes),
        p_prev=jnp.array(0.0),
    )

def ace_reset_traces(state):
    """
    run between trials - 
    reset x_bar and p_prev, but keep v (so we keep learning!)
    """
    return state._replace(
        x_bar=jnp.zeros_like(state.x_bar),
        p_prev=jnp.array(0.0),
    )

def ace_get_r_hat(state, box_vec, r, gamma):
    """
    get r_hat from ACE; also returns p because we need it for the update
    """
    p = state.v @ box_vec # (4)
    r_hat = r + gamma * p - state.p_prev # (7)
    return r_hat, p


def ace_update(state, box_vec, r_hat, p, beta, lam):
    """
    update ACE params
    """
    v_new = state.v + beta * r_hat * state.x_bar # (5)

    x_bar_new = lam * state.x_bar + (1 - lam) * box_vec # (6)

    return ACEState(v=v_new, x_bar=x_bar_new, p_prev=p)
