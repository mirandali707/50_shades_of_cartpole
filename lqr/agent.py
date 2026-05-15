from typing import NamedTuple
import numpy as np
from scipy.linalg import solve_discrete_are


model_name = "LQR"

SAVE_PATH = "lqr_best.npz"


class LQRAgent(NamedTuple):
    """
    linear quadratic regulator! K is the (1,4) gain matrix (1 input, 4 observations)
    """
    K: np.ndarray


# constants from gymnasium's cartpole implementation
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
LENGTH = 0.5 # actually half length but who's counting
FORCE_MAG = 10.0
TAU = 0.02 # euler step


def _build_cartpole_AB():
    """
    linearize our dynamics around upright fixed point (theta=0, theta_dot=0)
    and Euler-discretize with dt = TAU
    ret (A, B) so that x_{t+1} = A x_t + B F_t.
    """
    m, M, l, g, dt = MASS_POLE, MASS_CART, LENGTH, GRAVITY, TAU
    m_t = M + m
    d = l * (4.0 / 3.0 - m / m_t) # linearized denominator from theta_ddot eqn

    # continuous-time A_c, B_c
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

    # Euler discretization (gymnasium does euler, so we do too!)
    A = np.eye(4) + A_c * dt
    B = B_c * dt
    return A, B


def _solve_lqr(A, B, Q, R):
    """
    discrete infinite-horizon LQR
    returns gain K such that u = -K x is optimal
    for cost sum_t x'Qx + u'Ru.
    I do not totally understand what is going on here.
    """
    P = solve_discrete_are(A, B, Q, R)
    # K = (R + B' P B)^-1 B' P A
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def init_agent():
    A, B = _build_cartpole_AB()
    Q = np.diag([1.0, 1.0, 10.0, 10.0])
    R = np.array([[1.0]])
    K = _solve_lqr(A, B, Q, R)
    return LQRAgent(K=K)


def get_action(state, observation, act_key):
    """
    get control input u = -K x then threshold to discrete action
    """
    x = np.asarray(observation, dtype=np.float64)
    u = float((-state.K @ x)[0]) # (1,1) matrix -> scalar
    action = 1 if u > 0.0 else 0
    return action


def no_op(state, *args, **kwargs):
    return state

reset_agent = no_op
update = no_op