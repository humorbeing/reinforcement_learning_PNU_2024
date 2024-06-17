import gymnasium as gym


env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
for _ in range(20):
    # env.render()
    action = env.action_space.sample()
    print(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

print('')