from __future__ import print_function
import tensorflow as tf
import numpy as np

#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3
# #create tensorflow structure start
# # 随机 1维 数据范围 -1.0-1.0 tf变量
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))

biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
# L2
loss=tf.reduce_mean(tf.square(y-y_data))
#学习效率0.5
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)


#create tensorflow structure end

sess=tf.Session()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)


for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))