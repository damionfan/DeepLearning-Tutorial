import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
max_steps=1000
lr=1e-3
dropout=0.9
data_dir='/tmp/tensorflow/mnist/input_data'
log_dir='/tmp/tensorflow/mnist/logs/mnist_with_summaries'
mnist=input_data.read_data_sets(data_dir,one_hot=True)

with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y_=tf.placeholder(tf.float32,[None,10],name='y-input')

with tf.name_scope('input_reshape'):
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)#max_image=10

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))
'''计算variable的mean，stddev,max,min然后使用tf.summary.scalar'''
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min'.tf.reduce_min(var))
        tf.summary.histogram('histogram',var)


def nn_layer(input_tensor,input_dim,out_dim,layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights=weight_variable([input_dim,output_dim])
        with tf.name_scope('biases'):


