import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

sess=tf.InteractiveSession()

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def con2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])


x_image=tf.reshape(x,[-1,28,28,1])



#占位符
x =tf.placeholder('float32',shape=[None,784])
y_=tf.placeholder('float32',shape=[None,10])#每一行是10维的one-hot

#变量 变量需要session初始化
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.maxmul(x,W)+b)
cross_entropy=-tf.reduce_sum(y_*tf.log(y))


train_step=tf.train.GradientDescentOpitimizer(0.01).minimize(cross_entropy)



sess.run(tf.global_variables_initializer())

for i in range(100):
    batch=mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))
print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))