import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

env = gym.make(
    "CartPole-v1",
    render_mode="none",
)

N_TRIALS = 1000
N_REPS = 10

# prng key
key = jax.random.PRNGKey(0)

def train_loop(
    init_agent,
    reset_agent,
    get_action,
    update,
    model_name,
    save_trained_agent,
):
    all_trial_results = []
    best_w = None
    best_reward = -1

    for rep in range(N_REPS):
        key = jax.random.fold_in(jax.random.PRNGKey(0), rep)
        state = init_agent()
        trial_results = []

        pbar = tqdm(range(N_TRIALS), desc=f"repeat {rep}")
        for trial_idx in pbar:
            observation, info = env.reset()
            state = reset_agent(state)
            
            # snapshot the w being used for this episode, before it runs
            # in case it does really good and we want to save it
            episode_w = state.curr_w

            episode_over = False
            steps = 0
            while not episode_over:
                key, act_key = jax.random.split(key)
                action = get_action(state, observation, act_key)
                prev_observation = observation
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                state = update(state, prev_observation, action, observation, reward, terminated, truncated, info)
                episode_over = terminated or truncated

            # episode is done — `steps` is the true reward for episode_w
            if steps > best_reward:
                best_reward = steps
                best_w = episode_w

            trial_results.append(steps)
            pbar.set_postfix({"steps": steps, "best": best_reward})
        
        pbar.close()
        all_trial_results.append(trial_results)

    env.close()
    all_trial_results = np.array(all_trial_results)
    np.save(f"{model_name}_100_trial_results.npy", all_trial_results)
    save_trained_agent(best_w)
    return all_trial_results