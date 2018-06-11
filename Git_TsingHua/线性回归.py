#liner regression

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng=np.random

#Parameters
learning_rate=0.01
training_epochs=1000
display_step=50

#Training Data
train_X=np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y=np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples=train_X.shape[0]#第一维度长

#TF placeholder
X=tf.placeholder('float')
Y=tf.placeholder('float')

#set model weights
W=tf.Variable(np.random.randn(),name='weights')
b=tf.Variable(np.random.randn(),name='bias')

#Construct a liner model
pred=tf.multiply(W,X)+b

#Loss
loss=tf.reduce_sum(tf.square(pred-Y))/(2*n_samples)

#Optimizer
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init=tf.global_variables_initializer()

#train
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        #for(x,y) in zip(train_X,train_Y):
        sess.run(optimizer,feed_dict={X:train_X,Y:train_Y})

        #Display
        if epoch+1 % display_step ==0:
            c,weights,bias=sess.run([loss,W,b],feed_dict={X:train_X,Y:train_Y})
            print('epoch:{0} loss={1} W={2} b={3}'.format(epoch,c,weights,bias))

    print("Train Finished")
    c, weights, bias = sess.run([loss, W, b], feed_dict={X: train_X, Y: train_Y})
    print('Trained loss={0} W={1} b={2}'.format( c, weights, bias))

    #Graphic display
    plt.plot(train_X,train_Y,'ro',label="Original Data")
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label="Fitted line")
    plt.legend()
    plt.show()

    #Testing example as requested
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    plt.close('all')
