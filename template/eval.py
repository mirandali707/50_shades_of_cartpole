import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

env = gym.make(
    "CartPole-v1",
    render_mode="none",
)

N_EVAL_TRIALS = 100
N_REPS = 10

# prng key
key = jax.random.PRNGKey(0)

def eval_loop(
    init_trained_agent, 
    reset_agent, 
    get_action, 
    model_name
):
    all_trial_results = []

    for rep in range(N_REPS):
        # independent key per rep, derived from rep index via fold_in
        key = jax.random.fold_in(jax.random.PRNGKey(0), rep)

        # TODO
        state = init_trained_agent()

        trial_results = []

        pbar = tqdm(range(N_EVAL_TRIALS), desc=f"repeat {rep}")
        for trial_idx in pbar:
            observation, info = env.reset()

            # TODO optionally reset the agent
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

                episode_over = terminated or truncated

            trial_results.append(steps)
            pbar.set_postfix({"steps": steps})
        
        pbar.close()
        all_trial_results.append(trial_results)

    env.close()

    all_trial_results = np.array(all_trial_results)  # shape (N_REPS, N_TRIALS)
    np.save(f"{model_name}_100_eval_results.npy", all_trial_results)

    # Compute average + std steps alive over the last K trials across reps
    mean_final = all_trial_results.mean()
    std_final = all_trial_results.std()
    print(f"Final trial (#{N_EVAL_TRIALS}) steps alive across {N_REPS} reps: "
        f"{mean_final:.2f} ± {std_final:.2f}")
    return all_trial_results