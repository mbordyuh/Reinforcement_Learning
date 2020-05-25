"""
Solve cartpole problem with random weight search
"""
import gym
import numpy as np


def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0

    while True:
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, _ = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def draw(env, parameters):
    observation = env.reset()
    for _ in range(500):
        env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, _ = env.step(action)

def random_search(env):
    bestreward = 0

    for i in range(10000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > bestreward:
            bestreward = reward
            # considered solved if the agent lasts 200 timesteps
            if reward == 200:
                break
    print('Number of episodes to find a solution:', i)
    draw(env, parameters)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    random_search(env)
