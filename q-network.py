import gym
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_eager_execution()

env = gym.make('FrozenLake-v0')

# Parameters
total_states = 16
input_size = env.observation_space.n
output_size = env.action_space.n
lr = 0.1

def one_hot(x):
    return np.identity(total_states)[x:x+1]

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([input_size, int(input_size//2)],0,0.01))
W2 = tf.Variable(tf.random_uniform([int(input_size//2), output_size],0,0.01))

Q1   = tf.matmul(X,W1)
Qpred = tf.matmul(Q1, W2)
Y       = tf.placeholder(shape=[1,output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))

train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

dis = .99
num_ep = 4000

rList = []
init = tf.global_variables_initializer()

# Training code
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_ep):
        s = env.reset()
        e = 1./((i/50)+10)
        rAll = 0
        done = False
        local_loss = []

        while not done:
            Qs = sess.run(Qpred, feed_dict={X:one_hot(s)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1, reward, done, _ = env.step(a)
            if done:
                Qs[0, a]=reward
            else:
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
                Qs[0,a] = reward + dis*np.max(Qs1)

            sess.run(train, feed_dict={X: one_hot(s), Y:Qs})

            rAll += reward
            s = s1
        rList.append(rAll)
plt.plot(rList)

print("Percent of successful eps: ", str(sum(rList)*100/num_ep)+"%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()