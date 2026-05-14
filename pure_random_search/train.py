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
    save_trained_agent
):
    all_trial_results = []
    curr_best_model = None
    curr_best_reward = -1

    for rep in range(N_REPS):
        # independent key per rep, derived from rep index via fold_in
        key = jax.random.fold_in(jax.random.PRNGKey(0), rep)

        # initialize model
        state = init_agent()

        trial_results = []

        pbar = tqdm(range(N_TRIALS), desc=f"repeat {rep}")
        for trial_idx in pbar:
            observation, info = env.reset()

            # optionally reset the agent
            state = reset_agent(state)

            episode_over = False
            steps = 0

            while not episode_over:
                # this is the nice jax way to get a new random seed which is properly independent
                key, act_key = jax.random.split(key)

                # get action (use act_key if it is random)
                action = get_action(state, observation, act_key)

                prev_observation = observation

                # execute action in env
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1

                # do something with the new info
                state = update(state, prev_observation, action, observation, reward, terminated, truncated, info)

                episode_over = terminated or truncated

            trial_results.append(steps)
            pbar.set_postfix({"steps": steps})
        
        pbar.close()

        # model checkpoint - save best-performing model
        # NOTE this is different from the RL methods, we directly compare the episodic reward of the best w
        if state.best_reward > curr_best_reward:
            curr_best_model = state
            curr_best_reward = state.best_reward
        all_trial_results.append(trial_results)

    env.close()

    all_trial_results = np.array(all_trial_results)  # shape (N_REPS, N_TRIALS)
    np.save(f"{model_name}_100_trial_results.npy", all_trial_results)

    
    save_trained_agent(curr_best_model)
    return all_trial_results