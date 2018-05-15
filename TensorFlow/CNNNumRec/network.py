import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuary(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
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
x_image=tf.reshape(xs,[-1,28,28,1])#-1:不管他的维度，之后再加。1：channel 黑色就是1
# print(x_image.shape)#[n_samples,28,28,1]


#-----conv1 layer
W_conv1=weight_variable([5,5,1,32])#5*5 ,输入厚度=1,输出厚度=32
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#28*28*32
h_pool1=max_pool_2x2(h_conv1)#output size 14*14*32
#-----conv2 layer
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)#7*7*64
#-----func1 layer
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#[n_sample,7,7,64]->[n_sampel,7*7*64]
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#-----func2 layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


#loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#train
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50==0:
        print(compute_accuary(mnist.test.images,mnist.test.labels))
