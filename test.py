import d4rl
import gym
import mujoco_py

import gym

env = gym.make("door-v0")
observation = env.reset()
while True:
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    env.render()

    # if done:
    # observation = env.reset()
env.close()
