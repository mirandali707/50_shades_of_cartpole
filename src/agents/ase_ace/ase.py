from typing import NamedTuple
import jax
import jax.numpy as jnp

class ASEState(NamedTuple):
    """
    Associative Search Element;
    produces control action from box vector + reinforcement signal (external or internal)
    """
    w: jnp.ndarray # (n_boxes,)
    e: jnp.ndarray # (n_boxes,)

def ase_init(n_boxes):
    """
    initialize an all-zero starting state for ASE
    """
    return ASEState(w=jnp.zeros(n_boxes), e=jnp.zeros(n_boxes))

def ase_reset_traces(state):
    """
    run between trials - 
    reset e, but keep v (so we keep learning!)
    """
    return state._replace(
        e=jnp.zeros_like(state.e),
    )

def ase_get_action(state, box_vec, seed, sigma):
    """
    takes current ASE state, box_vec for our current obs, random seed, and sigma (std of noise)
    returns action as per eqn (1)
    """
    noise = jax.random.normal(seed) * sigma
    inner = state.w @ box_vec + noise
    # f(x)
    return jnp.where(inner >= 0, 1.0, -1.0)  # 1 if inner is >= 0, otherwise -1

def ase_update(state, last_x, last_y, r_hat, alpha, delta):
    """
    takes ASE state, previous x, previous y, internal reward, and hyperparams alpha, delta
    updates ASE weights as per eqns (2), (3)
    """
    w_new = state.w + alpha * r_hat * state.e # eqn (2)
    e_new = delta * state.e + (1 - delta) * last_y * last_x # eqn (3)
    return ASEState(w=w_new, e=e_new)
