import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time
import math
import os

epochs=3000
batch_size=128
num_examples = 10000
lr=1e-3
data_dir='./tmp/cifar10_data/cifar-10-batches-bin'
cifar10.maybe_download_and_extract()#下载解压


def weight_with_loss(shape,stddev,w=None):#w1控制L2正则化的大小
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w,name='weight_loss')
        tf.add_to_collection('losses',weight_loss)#add to losses[]
    return var
def bias(shape,value=0):
    return tf.Variable(tf.constant(value,shape=shape,dtype=tf.float32))

def conv_op(x,w,strides=[1,1,1,1],padding="SAME"):
    return tf.nn.conv2d(x,w,strides=strides,padding=padding)
def pool_op(x,ksize=[1,2,2,1],strides=[1,2,2,1]):
    return tf.nn.max_pool(x,ksize=ksize,strides=strides,padding="SAME")
def loss_op(logits,labels):
    labels=tf.cast(labels,tf.int64)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name="cross_entropy")
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss_cross_entropy_and_l2loss')
'''Args:列表元素相加
inputs:
A list of Tensor objects, each with same shape and type.
name: A name for the operation (optional).
Returns:
A Tensor of same shape and type as the elements of inputs.'''


images_train,labels_train=cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)#数据增强，水平翻转，随机剪贴24*24，随机亮度和对比度，数据标准化 ,多线程加速
images_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)#裁剪中间24*24 数据标准化

data_image=tf.placeholder(tf.float32,[batch_size,24,24,3])
data_label=tf.placeholder(tf.int32,[batch_size])

#convolutional layer
with tf.name_scope("conv1_pool1_lrn_1"):
    filter1=weight_with_loss([5,5,3,64],stddev=0.05)
    bias1=bias([64])
    conv1=tf.nn.relu(conv_op(data_image,filter1)+bias1)
    pool1=pool_op(conv1,ksize=[1,3,3,1],strides=[1,2,2,1])
    lrn1=tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)#lrn......

with tf.name_scope('conv2_lrn2_pool2'):
    filter2=weight_with_loss([5,5,64,64],stddev=0.05)
    bias2=bias([64],0.1)
    conv2=tf.nn.relu(conv_op(lrn1,filter2)+bias2)
    lrn2=tf.nn.lrn(conv2,4,bias=1,alpha=0.001/9.0,beta=0.75)
    pool2=pool_op(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1])

#flatten
with tf.name_scope('flatten'):
    flatten=tf.reshape(pool2,[batch_size,-1])
    dim=flatten.get_shape()[1].value#get dimension

#FC
with tf.name_scope("FC1_l2Loss"):
    w1=weight_with_loss([dim,384],stddev=0.04,w=0.04)
    b1=bias([384],0.1)
    fc1=tf.nn.relu(b1+tf.matmul(flatten,w1))
with tf.name_scope("FC2_l2Loss"):
    w2=weight_with_loss([384,192],stddev=0.04,w=0.04)
    b2=bias([192],0.1)
    fc2=tf.nn.relu(b2+tf.matmul(fc1,w2))
with tf.name_scope('Predicton_logits'):
    w3=weight_with_loss([192,10],1.0/192)
    b3=bias([10],0)
    logits=tf.add(tf.matmul(fc2,w3),b3)
#inference finished

loss=loss_op(logits,data_label)
tf.summary.scalar("loss",loss)

train=tf.train.AdamOptimizer(lr).minimize(loss)

top_k_op=tf.nn.in_top_k(logits,data_label,1)#return bool,

merged=tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('cifar_log',sess.graph)
    tf.train.start_queue_runners(sess)#数据增强的线程加速队列，16个线程
    s_time=time.time()
    for epoch in range(epochs):
        start_time=time.time()
        batch=sess.run([images_train,labels_train])
        sess.run(train,feed_dict={data_image:batch[0],data_label:batch[1]})
        duration=time.time()-start_time
        if epoch%10 ==0:
            print('Sequential_examples_demo/s'%(duration/batch_size),end='')
            print('second/batch'%float(duration))
            res=sess.run(merged,feed_dict={data_image:batch[0],data_label:batch[1]})
            writer.add_summary(res,epoch)

    e_time=time.time()
    print("train time"%float(e_time-s_time))
    num_iter=int(math.ceil(num_examples/batch_size))#ceil 向上取整
    true_count=0
    total_sample_count=num_iter*batch_size
    step=0
    while step<num_iter:
        batch=sess.run([images_test,labels_test])
        pred=sess.run([top_k_op],feed_dict={data_image:batch[0],data_label:batch[1]})
        true_count+=np.sum(pred)
        step+=1
    precision=true_count/total_sample_count
    print('test precision:'%precision)