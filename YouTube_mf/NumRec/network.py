import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#add layer
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

#define placeholder
#这里的shape？？？:第一个代表batch不定 第二个是平铺图像
xs=tf.placeholder(tf.float32,[None,784])#28*28
ys=tf.placeholder(tf.float32,[None,10])

#add output layer
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)

#loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#define accuary
#---------------------tf 函数--------------------#
def compute_accuary(v_xs,v_ys):
    global prediction
    y_pre =sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuary=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuary,feed_dict={xs:v_xs,ys:v_ys})
    return result




sess=tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuary(mnist.test.images,mnist.test.labels))