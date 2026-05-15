import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import jax


def record_video(
    init_trained_agent,
    reset_agent,
    get_action,
    model_name,
    video_dir="",
):
    # rgb_array render mode is required for RecordVideo to capture frames
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=model_name,
        episode_trigger=lambda ep: True,  # record every episode (we only run one)
    )

    key = jax.random.PRNGKey(0)
    state = init_trained_agent()
    state = reset_agent(state)

    observation, info = env.reset()
    episode_over = False
    steps = 0

    while not episode_over:
        key, act_key = jax.random.split(key)
        state, action = get_action(state, observation, act_key)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        episode_over = terminated or truncated

    env.close()
    print(f"recorded {steps}-step rollout to {video_dir}/{model_name}-episode-0.mp4")