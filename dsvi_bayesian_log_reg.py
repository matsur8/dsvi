# -*- encoding:-utf-8 -*-

import numpy as np
import scipy.special as spsp
import tensorflow as tf

N = 1000
M = 10
T = 20

tf.set_random_seed(12)
np.random.seed(24)

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

mu = tf.Variable(tf.zeros(M))
sigma0 = tf.Variable(tf.ones(M))
#sigma = tf.exp(sigma0)
#sigma = tf.abs(sigma0)
sigma = sigma0
e = tf.random_normal([M])
z = e * sigma + mu

std_multi_normal = tf.contrib.distributions.MultivariateNormalDiag([0.0]*M)
log_p_z = std_multi_normal.log_prob(z)
log_p_d_given_z = - tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(X,tf.reshape(z, [-1,1]))))

dist_q = tf.contrib.distributions.MultivariateNormalDiag(mu, sigma) #negative scales are allowed.
entropy_q = tf.contrib.bayesflow.entropy.entropy_shannon(dist_q)

loss = (- entropy_q - log_p_d_given_z - log_p_z) / N
optimizer = tf.train.GradientDescentOptimizer(1.0)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

X_data = np.random.normal(size=(N,M))
w_true = np.random.normal(size=M)
y_data = np.random.binomial(1, spsp.expit(X_data.dot(w_true))).reshape(-1,1)

for t in range(T):
   grad = optimizer.compute_gradients(loss, [mu, sigma0])
   update = optimizer.apply_gradients(grad)
   sess.run(update, {X:X_data, y:y_data})
   #print(sess.run(log_p_d_given_z, {X:X_data,y:y_data}))
   print("epoch:{}, loss:{}".format(t+1,sess.run(loss, {X:X_data,y:y_data})))


print(w_true)
print(sess.run(mu))
print(sess.run(sigma))

