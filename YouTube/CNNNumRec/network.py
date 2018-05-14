import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuary(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction),tf.float32)
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#----------------define----------------
def weight_variable(shape):
    initial =tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial= tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):#[1,x_move,y_move,1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


#define placeholder
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

#-----conv1 layer
#-----conv2 layer
#-----func1 layer
#-----func2 layer


#loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#train
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuary(minist.test.images,mnist.test.labels))
