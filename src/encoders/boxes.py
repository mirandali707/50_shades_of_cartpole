import jax.numpy as jnp

N_BOXES = 162

def decoder(observation):
    """
    takes in state (x, x_dot, theta, theta_dot) and discretizes into 162 bins as specified in the paper, 
    from Michie & Chambers' boxes system
    """
    x, x_dot, theta, theta_dot = observation

    # (3) x: +/- 0.8, +/- 2.4
    if x < -0.8:
        x_bin = 0
    elif x < 0.8:
        x_bin = 1
    else:
        x_bin = 2
    
    # (6) theta: 0, +/- 1, +/- 6, +/- 12
    theta_deg = jnp.degrees(theta)  # paper specifies thresholds in degrees
    if theta_deg < -6:
        theta_bin = 0
    elif theta_deg < -1:
        theta_bin = 1
    elif theta_deg < 0:
        theta_bin = 2
    elif theta_deg < 1:
        theta_bin = 3
    elif theta_deg < 6:
        theta_bin = 4
    else:
        theta_bin = 5
    
    # (3) x_dot: +/- 0.5, +/- inf
    if x_dot < -0.5:
        x_dot_bin = 0
    elif x_dot < 0.5:
        x_dot_bin = 1
    else:
        x_dot_bin = 2

    # (3) theta_dot: +/0 50, +/- inf
    theta_dot_deg = jnp.degrees(theta_dot)
    if theta_dot_deg < -50:
        theta_dot_bin = 0
    elif theta_dot_deg < 50:
        theta_dot_bin = 1
    else:
        theta_dot_bin = 2

    # convert into idx of 3 * 6 * 3 * 3 = 162 bins
    box_idx = (
        x_bin * (3 * 6 * 3)
        + x_dot_bin * (6 * 3)
        + theta_bin * 3
        + theta_dot_bin
    )
    arr = jnp.zeros(N_BOXES)
    arr = arr.at[box_idx].set(1)
    return arr