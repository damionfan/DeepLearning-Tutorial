import tensorflow as tf
import numpy as np
#liner regression
x=np.random.rand(100).astype(np.float32)
y=x*0.1+0.3

w = tf.Variable(tf.random_uniform([1],-1.0,1.0))

b = tf.Variable(tf.zeros([1]))

# w=tf.Variable(0.0,name='weights')
# b=tf.Variable(0.0,name='bias')

y_pred=x*w+b

loss=tf.reduce_mean(tf.square(y-y_pred))

optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Writer=tf.summary.FileWriter('./graphs',graph=sess.graph)
    for i in range(200):
        sess.run(optimizer)
        if i%20==0 :
            print(sess.run(w),sess.run(b))

'''
list of optimizers in TF
tf.train.GradientDescentOptimizer()
tf.train.AdagradOptimizer()
tf.train.MomentumOptimizer()
tf.train.AdamOptimizer()
tf.train.ProximalGradientDescentOptimizer()
tf.train.ProximalAdagradOptimizer()
tf.train.RMSPropOptimizer()
'''
'''huber loss
def huber_loss(labels,predictions,delta=1.0):
    residual=tf.abs(predictions-labels)
    condition=tf.less(residual,delta)
    small_res=0.5*tf.square(residual)
    large_res=delta*residual-0.5*tf.square(delta)
    return tf.select(condition,small_res,large_res)
'''