import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


x_data=np.asarray([338,333,328,207,226,25,179,60,208,606])
y_data=np.asarray([640,633,619,393,428,27,193,66,226,1591])

learning_rate=0.0000001
epochs=1000

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.zeros([1]))# tf.constant() takes a NumPy array as an argument

pred=tf.add(b,tf.multiply(W,X))
loss=tf.reduce_mean(tf.square(Y-pred))

optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train=optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        sess.run(train,feed_dict={X:x_data,Y:y_data})
    print(sess.run(W),sess.run(b),sess.run(loss,feed_dict={X:x_data,Y:y_data}))
    ''' 出现了nan之前 是因为learning-rate 太高了'''
    plt.plot(x_data,y_data,'ro',label="original data")
    plt.plot(x_data,sess.run(pred,feed_dict={X:x_data,Y:y_data}),label="pred")

    plt.legend()
    plt.show()
    plt.close()
