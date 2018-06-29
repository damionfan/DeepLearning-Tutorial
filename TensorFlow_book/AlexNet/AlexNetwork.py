from datetime import datetime
import math
import time
import tensorflow as tf

batch_size=32
num_batch=100

def print_activation(t):
    #print name and shape
    print(t.op.name," ",t.get_shape().as_list())

def inference(images):
    parameters=[]

    with tf.name_scope("conv1") as scope:
        kernel=tf.variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(image,kernel,[1,4,4,1],padding="SAME")
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activation(conv1)
        parameters+=[kernel,biases]

