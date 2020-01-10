import gym
import numpy as np
from gym.envs.registration import register
import sys, tty, readchar
import random
import matplotlib.pyplot as plt

def rargmax(vector) :
    m = np.amax(vector)                    # Return the maximum of an array or maximum along an axis (0 아니면 1)
    indices = np.nonzero(vector == m)[0]   # np.nonzero(True/False vector) => 최대값인 요소들만 걸러내
    return random.choice(indices)          # 그 중 하나 랜덤으로 선택

register(
    id="FrozenLake-v3",
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery':True}
)

env = gym.make('FrozenLake-v0')
env.render()

Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000
dis = 0.99
lr = 0.85 # 고집

rList=[] # 얼마나 잘 했는지 각 Episode 마다의 result를 기록하는 List.
for i in range(num_episodes):

    e = 1. /((i/100)+1)

    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state, :]+np.random.randn(1, env.action_space.n)/(i+1))
        
        new_state, reward, done, _ = env.step(action)

        Q[state,action] = (1.0-lr)*Q[state,action] + lr*(reward + dis*np.max(Q[new_state, :]))
        
        rAll += reward
        state = new_state
    
    rList.append(rAll)

# Result reporting
print("Sucess rate: ", str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT\t\tDOWN\t\tRIGHT\t\tUP")
print(Q)

plt.bar(range(len(rList)),rList,color="blue")
plt.show()