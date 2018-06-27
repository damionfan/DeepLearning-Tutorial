import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

'''
x=tf.placeholder(tf.float32,[batch_size,784],name)
y=tf.placeholder(tf.float32,[batch_size,10],name)
'''
learning_rate=0.01
batch_size=128
n_epochs=30
#step1 read data
MNIST=input_data.read_data_sets('/data/mnist',one_hot=True)
#step2 create placeholder
X=tf.placeholder(tf.float32,[batch_size,784],name='x_placeholder')
Y=tf.placeholder(tf.float32,[batch_size,10],name='y_placeholder')
#step3 create weights bias
w=tf.Variable(tf.random_normal([784,10]),name='weights')
b=tf.Variable(tf.zeros([1,10]),name='bias')
#step4 build model
logits=tf.matmul(X,w)+b
#step5 define loss function
entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='loss')
loss=tf.reduce_mean(entropy)
#step6 define train op
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    writer=tf.summary.FileWriter('./graphs',sess.graph)

    start_time=time.time()
    sess.run(tf.global_variables_initializer())
    n_batchs=int(MNIST.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss=0
        for _ in range(n_batchs):
            X_batch,Y_batch=MNIST.train.next_batch(batch_size)
            _,loss_batch=sess.run([optimizer,loss],feed_dict={X:X_batch,Y:Y_batch})
            total_loss+=loss_batch
        print("average loss epoch:{0}:{1}".format(i,total_loss/n_batchs))

    print("total time : {0} secends".format(time.time()-start_time))
    print('Optimizer Finished!')

    #test model
    n_batchs=int(MNIST.test.num_examples/batch_size)
    total_correct_preds=0
    for i in range(n_batchs):
      X_batch,Y_batch=MNIST.test.next_batch(batch_size)#下面在优化！ 过拟合
      _,loss_batch,logits_batch=sess.run([optimizer,loss,logits],feed_dict={X:X_batch,Y:Y_batch})
      preds=tf.nn.softmax(logits_batch)
      correct_preds=tf.equal(tf.argmax(preds,1),tf.argmax(Y_batch,1))
      accuracy=tf.reduce_sum(tf.cast(correct_preds,tf.float32))
      total_correct_preds+=sess.run(accuracy)
    print("accuracy :{0}".format(total_correct_preds/MNIST.test.num_examples))

    writer.close()
