import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

epochs=100
batch_size=100
display_steps=1 #间隔
learning_rate=1e-4
keep_prob=0.5
fch_nodes=512
'''
输入层为输入的灰度图像尺寸:  -1 x 28 x 28 x 1 
第一个卷积层,卷积核的大小,深度和数量 (5, 5, 1, 16)
池化后的特征张量尺寸:       -1 x 14 x 14 x 16
第二个卷积层,卷积核的大小,深度和数量 (5, 5, 16, 32)
池化后的特征张量尺寸:       -1 x 7 x 7 x 32
全连接层权重矩阵         1568 x 512
输出层与全连接隐藏层之间,  512 x 10'''

def weight_init(shape):
    weights=tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights)
def bias_init(shape):
    bias=tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(bias)
def get_random_batchdata(n_samples,batchsize):
    start_index=np.random.randint(0,n_samples-batchsize)
    return (start_index,start_index+batchsize)
def xavier_init(layer1,layer2,constant=1):#全连接层权重初始化函数xavier
    Min=-constant*np.sqrt(6.0/(layer1+layer2))
    Max=constant*np.sqrt(6.0/(layer1+layer2))
    return tf.Variable(tf.random_uniform((layer1,layer2),minval=Min,maxval=Max))

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

x_image=tf.reshape(x,[-1,28,28,1])

#model
w_conv1=weight_init([5,5,1,16])
b_conv1=bias_init([16])
conv1=tf.nn.relu(tf.nn.conv2d(x_image,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
pool1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='SAME')

w_conv2=weight_init([5,5,16,32])
b_conv2=bias_init([32])
conv2=tf.nn.relu(tf.nn.conv2d(pool1,w_conv2,strides=[1,1,1,1],padding='SAME'))
pool2=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding='SAME')

flatten=tf.reshape(pool2,[-1,7*7*32])

w_fc1=xavier_init(7*7*32,fch_nodes)
b_fc1=bias_init([fch_nodes])
fc1=tf.nn.relu(tf.matmul(flatten,w_fc1)+b_fc1)
drop1=tf.nn.dropout(fc1,keep_prob=keep_prob)

w_fc2=xavier_init(fch_nodes,10)
b_fc2=bias_init([10])
y_=tf.add(tf.matmul(drop1,w_fc2),b_fc2)

y_out=tf.nn.softmax(y_)

# cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y_,y) not mean
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_out),reduction_indices=1))

correct_predictin=tf.equal(tf.argmax(y,1),tf.argmax(y_out,1))
#bool -> float
accuracy=tf.reduce_mean(tf.cast(correct_predictin,tf.float32))

trian_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init=tf.global_variables_initializer()
MNIST=input_data.read_data_sets('MNIST/mnist',one_hot=True)

with tf.Session() as sess:
    sess.run(init)
    Loss=[]
    Accuracy=[]
    for i in range(epochs):
        X_batch, Y_batch = MNIST.train.next_batch(batch_size)
        _,loss,accu=sess.run([trian_op,cross_entropy,accuracy],feed_dict={x:X_batch,y:Y_batch})

        Loss.append(loss)
        Accuracy.append(accu)

        if i % display_steps==0:
            pritn('epoch : %d ,loss : %.7f'%(i+1,loss))

    print('train fk done')
    fig1,ax1=plt.subplots(figsize=(10,7))
    plt.plot(Loss)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.title('Loss')
    plt.grid()
    plt.show()

    fig7,ax7=plt.subplots(figsize=(10,7))
    plt.plot(Accuracy)
    ax7.set_xlabel('Epochs')
    ax7.set_ylabel('acc rate')
    plt.title('Train acc rate')
    plt.grid()
    plt.show()

    '''-------------------------------------------------------------'''
    fig2,ax2=plt.subplots(figsize=(2,2))
    ax2.imshow(np.reshape(MNIST.train.images[np.random.choice(100,size=1)],(28,28)))
    plt.show()

    #conv1
    input_image =MNIST.train.images[11:12]
    conv1_16=sess.run(conv1,feed_dict={x:input_image})#[1,28,28,16]
    conv1_transpose=sess.run(tf.transpose((conv1_16,[3,0,1,2])))
    fig3,ax3=plt.subplots(nrows=1,ncols=16,figsize=(16,1))
    for i in range(16):
        ax3[i].imshow(conv1_transpose[i][0])
        # ax3[i].imshow(np.reshape(conv1_16,(16,1,28,28))[i][0])

    plt.title('pool1 16*28*28')
    # 第一层池化后的特征图
    pool1_16 = sess.run(h_pool1, feed_dict={x: input_image})  # [1, 14, 14, 16]
    pool1_transpose = sess.run(tf.transpose(pool1_16, [3, 0, 1, 2]))
    fig4, ax4 = plt.subplots(nrows=1, ncols=16, figsize=(16, 1))
    for i in range(16):
        ax4[i].imshow(pool1_transpose[i][0])

    plt.title('Pool1 16x14x14')
    plt.show()

    # 第二层卷积输出特征图
    conv2_32 = sess.run(h_conv2, feed_dict={x: input_image})  # [1, 14, 14, 32]
    conv2_transpose = sess.run(tf.transpose(conv2_32, [3, 0, 1, 2]))
    fig5, ax5 = plt.subplots(nrows=1, ncols=32, figsize=(32, 1))
    for i in range(32):
        ax5[i].imshow(conv2_transpose[i][0])
    plt.title('Conv2 32x14x14')
    plt.show()

    # 第二层池化后的特征图
    pool2_32 = sess.run(h_pool2, feed_dict={x: input_image})  # [1, 7, 7, 32]
    pool2_transpose = sess.run(tf.transpose(pool2_32, [3, 0, 1, 2]))
    fig6, ax6 = plt.subplots(nrows=1, ncols=32, figsize=(32, 1))
    plt.title('Pool2 32x7x7')
    for i in range(32):
        ax6[i].imshow(pool2_transpose[i][0])

    plt.show()