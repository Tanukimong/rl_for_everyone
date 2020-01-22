import gym
import numpy as np
from gym.envs.registration import register
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import time

# Parameters
register(
    id="FrozenLake-v3",
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery':True}
)

env = gym.make('FrozenLake-v3')

input_size = env.observation_space.n
output_size = env.action_space.n

print("Input size :{}, Output size :{}".format(input_size,output_size))

lr = 0.01

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_size),
    tf.keras.layers.Dense(output_size)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')

dis = .99
num_ep = 2000

rList = []

# Training code

start_time = time.time()

with tf.device("GPU:0"):
    for i in range(num_ep):

        print("Ep.{} Starting...".format(i))

        s = env.reset()
        e = 1./((i/50)+10)
        rAll = 0
        done = False
        while not done:
            X = tf.one_hot(s,input_size)
            X = tf.expand_dims(X,0)
            Qs = model.predict(X,verbose=0)

            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = tf.math.argmax(Qs,1)
            s1, reward, done, _ = env.step(int(a))
            if done:
                Qs[0, a]=reward
            else:
                X1 = tf.one_hot(s1,input_size)
                X1 = tf.expand_dims(X1,0)
                Qs1 = model.predict(X1,verbose=0)
                Qs[0,a] = reward + dis*tf.reduce_max(Qs1)
            
            X = tf.expand_dims(tf.one_hot(s,input_size),0)
            model.fit(X,Qs,verbose=0)

            rAll += reward
            s = s1
        rList.append(rAll)
        time_cs = time.time()-start_time
        print("From now = {:d}min {:d}sec\n".format(int(time_cs//60), int(time_cs%60)))


model.summary()

plt.plot(rList)

print("Percent of successful eps: ", str(sum(rList)/num_ep)+"%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()