import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

def filter_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))
def conv2d(x,filter):
    return tf.nn.conv2d(x,filter,strides=[1,1,1,1],padding='SAME')#x:4-D[batch,in_height,in_width,in_channel] filter[height,width,in_channel,out_channel]
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#strides[0]=[3]必须等于1
'''我对ksize的理解是filter'''
lr=1e-4#0.0001
#placeholder
with tf.name_scope('input'):
    data_x=tf.placeholder(tf.float32,[None,784])
    data_y=tf.placeholder(tf.float32,[None,10])
    x_image=tf.reshape(data_x,[-1,28,28,1])
keep_prob=tf.placeholder(tf.float32)

#model
#convolutional layer
with tf.name_scope('conv1_layer'):
    filter_conv1=filter_variable([5,5,1,32])#window,in_channel,out_channel
    bias_conv1=bias_variable([32])
    with tf.name_scope('conv1_pool1'):
        conv1=tf.nn.relu(conv2d(x_image,filter_conv1)+bias_conv1)
        pool1=max_pool_2x2(conv1)#map=14*14*32
with tf.name_scope('conv2_layer'):
    filter_conv2=filter_variable([5,5,32,64])
    bias_conv2=bias_variable([64])
    with tf.name_scope('conv2_pool2'):
        conv2=tf.nn.relu(conv2d(pool1,filter_conv2)+bias_conv2,name='conv2')
        pool2=max_pool_2x2(conv2)#7*7*64

#flatten
flatten=tf.reshape(pool2,[-1,7*7*64],name='flatten')

#fully connected layer
with tf.name_scope('FC_layer'):
    w_fc1=filter_variable([7*7*64,1024])
    b_fc1=bias_variable([1024])
    with tf.name_scope('FC1_drop'):
        fc1=tf.nn.relu(tf.add(b_fc1,tf.matmul(flatten,w_fc1)))
        drop1=tf.nn.dropout(fc1,keep_prob)
with tf.name_scope('Prediction'):
    w_fc2=filter_variable([1024,10])
    b_fc2=bias_variable([10])
    prediction=tf.nn.softmax(tf.matmul(drop1,w_fc2)+b_fc2)

#loss
with tf.name_scope('loss'):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(data_y*tf.log(prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',cross_entropy)

#train
train=tf.train.AdamOptimizer(lr).minimize(cross_entropy)

accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(data_y,1)),tf.float32))
tf.summary.scalar('accuracy',accuracy)

merged=tf.summary.merge_all()
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('./logs',sess.graph)
    for i in range(20000):
        batch=mnist.train.next_batch(50)
        fd_train={data_x:batch[0],data_y:batch[1],keep_prob:0.5}
        fd_tacc={data_x:batch[0],data_y:batch[1],keep_prob:1}
        fd_test={data_x:mnist.test.images,data_y:mnist.test.labels,keep_prob:1}
        sess.run(train,feed_dict=fd_train)
        if i% 100 ==0:
            res=sess.run(merged,feed_dict=fd_tacc)
            writer.add_summary(res,i)
    tacc=sess.run(accuracy,feed_dict=fd_test)
    print('test:acc',tacc)
