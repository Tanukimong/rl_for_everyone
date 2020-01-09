import gym
import numpy as np
from gym.envs.registration import register
import sys, tty, termios
import random

def rargmax(vector) :
    m = np.amax(vector)                    # Return the maximum of an array or maximum along an axis (0 아니면 1)
    indices = np.nonzero(vector == m)[0]   # np.nonzero(True/False vector) => 최대값인 요소들만 걸러내
    return random.choice(indices)          # 그 중 하나 랜덤으로 선택

register(
    id="FrozenLake-v3",
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery':False}
)

env = gym.make('FrozenLake-v3')
env.render()

Q = np.zeros([env.observation_space.n,env.observation_space.n])
num_episodes = 2000

rList=[]
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = rargmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        Q[state,action] = reward + np.max(Q[new_state, :])
        state = new_state