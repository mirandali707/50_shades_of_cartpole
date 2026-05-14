import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

env = gym.make(
    "CartPole-v1",
    render_mode="none",
)

N_TRIALS = 100
N_REPS = 10

# prng key
key = jax.random.PRNGKey(0)

K = 5

def train_loop(
    init_agent,
    reset_agent,
    get_action,
    update,
    model_name,
    save_trained_agent,
):
    all_trial_results = []
    curr_best_model = None
    curr_best_last_k = -1

    for rep in range(N_REPS):
        # independent key per rep, derived from rep index via fold_in
        key = jax.random.fold_in(jax.random.PRNGKey(0), rep)

        # TODO
        state = init_agent()

        trial_results = []

        pbar = tqdm(range(N_TRIALS), desc=f"repeat {rep}")
        for trial_idx in pbar:
            observation, info = env.reset()

            # TODO optionally reset the agent
            # NOTE added state as a param here so reset agent can also be pass thru
            state = reset_agent(state)

            episode_over = False
            steps = 0

            while not episode_over:
                # this is the nice jax way to get a new random seed which is properly independent
                key, act_key = jax.random.split(key)

                # TODO get action (use act_key if it is random)
                action = get_action(state, observation, act_key)

                prev_observation = observation

                # execute action in env
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1

                # TODO do something with the new info
                # NOTE updated this fn to take in prev_observation and action
                state = update(state, prev_observation, action, observation, reward, terminated, truncated, info)

                episode_over = terminated or truncated

            trial_results.append(steps)
            pbar.set_postfix({"steps": steps})
        
        pbar.close()

        avg_last_k = np.mean(trial_results[-K:])
        if avg_last_k > curr_best_last_k:
            curr_best_model = state
            curr_best_last_k = avg_last_k

        all_trial_results.append(trial_results)

    env.close()

    all_trial_results = np.array(all_trial_results)  # shape (N_REPS, N_TRIALS)
    np.save(f"{model_name}_100_trial_results.npy", all_trial_results)

    # Compute average + std steps alive over the last K trials across reps
    final_trial_steps = all_trial_results[:, -K]  # shape (N_REPS,K)
    mean_final = final_trial_steps.mean()
    std_final = final_trial_steps.std()
    print(f"Final trial (#{N_TRIALS}) steps alive across {N_REPS} reps: "
        f"{mean_final:.2f} ± {std_final:.2f}")
    save_trained_agent(curr_best_model)
    return all_trial_results