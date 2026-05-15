from typing import NamedTuple
import numpy as np
from scipy.optimize import minimize


model_name = "MPC"


# constants from gymnasium's cartpole implementation
GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
LENGTH = 0.5
FORCE_MAG = 10.0
TAU = 0.02

HORIZON = 20


class MPCAgent(NamedTuple):
    A: np.ndarray      # discrete dynamics
    B: np.ndarray
    Q: np.ndarray      # state cost
    R: np.ndarray      # input cost


def _build_cartpole_AB():
    """linearize around upright + Euler-discretize."""
    m, M, l, g, dt = MASS_POLE, MASS_CART, LENGTH, GRAVITY, TAU
    m_t = M + m
    d = l * (4.0 / 3.0 - m / m_t)

    A_c = np.array([
        [0.0, 1.0, 0.0,                  0.0],
        [0.0, 0.0, -(m * l * g) / (m_t * d), 0.0],
        [0.0, 0.0, 0.0,                  1.0],
        [0.0, 0.0, g / d,                0.0],
    ])
    B_c = np.array([
        [0.0],
        [(1.0 / m_t) * (1.0 + (m * l) / (m_t * d))],
        [0.0],
        [-1.0 / (m_t * d)],
    ])
    A = np.eye(4) + A_c * dt
    B = B_c * dt
    return A, B


def init_agent():
    A, B = _build_cartpole_AB()
    Q = np.diag([1.0, 1.0, 10.0, 10.0])
    R = np.array([[1.0]])
    return MPCAgent(A=A, B=B, Q=Q, R=R)


def _rollout_cost(u_seq, x0, A, B, Q, R):
    """
    given a candidate sequence of inputs, simulate forward with linear dynamics
    and accumulate the LQR cost sum_t (x'Qx + u'Ru).
    this is the objective scipy.optimize.minimize will minimize over u_seq.
    """
    x = x0.copy()
    cost = 0.0
    for u in u_seq:
        cost += x @ Q @ x + R[0, 0] * u * u
        x = A @ x + B[:, 0] * u
    cost += x @ Q @ x  # terminal state cost
    return cost


def get_action(state, observation, act_key):
    """
    replan at every step: minimize horizon-N cost over input sequence,
    apply first input, threshold to {0, 1}.
    """
    x0 = np.asarray(observation, dtype=np.float64)

    result = minimize(
        _rollout_cost,
        np.zeros(HORIZON), # initial guess all 0s
        args=(x0, state.A, state.B, state.Q, state.R),
        method="L-BFGS-B",
    )

    u0 = result.x[0]
    return 1 if u0 > 0.0 else 0


def no_op(state, *args, **kwargs):
    return state


reset_agent = no_op
update = no_op