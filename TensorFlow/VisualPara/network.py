import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


LR=0.1
REAL_PARAMS=[1.2,2.5]
INIT_PARAMS=[[5,4],
             [5,1],
             [2,4.5]][2]
x=np.linspace(-1,1,200,dtype=np.float32)

#test 1
y_fun=lambda a,b:a*x+b
tf_y_fun=lambda a,b:a*x+b

noise=np.random.rand(200)/10
y=y_fun(*REAL_PARAMS)+noise

a,b=[tf.Variable(initial_value=p,dtype=tf.float32)for p in INIT_PARAMS]
pred=tf_y_fun(a,b)
mse=tf.reduce_mean(tf.square(y-pred))
train_op=tf.train.GradientDescentOptimizer(LR).minimize(mse)

a_list,b_list,cost_list=[],[],[]
with tf.Session() as see:
    sess.run(global_variables_initializer())
    for t in range(400):
        a_,b_,mse_=sess.run([a,b,mse])
        a_list.append(a_);b_list.append(b_);cost_list.append(mse_)
        result,_=sess.run([pred,train_op])