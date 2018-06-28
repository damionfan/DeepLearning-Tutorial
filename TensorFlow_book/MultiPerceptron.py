import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


in_units=784
h1_units=300
lr=0.003

w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1),name='w1')
b1=tf.Variable(tf.zeros([h1_units]),name='b1')
w2=tf.Variable(tf.zeros([h1_units,10]),name='w2')
b2=tf.Variable(tf.zeros([10]),name='b2')

x=tf.placeholder(tf.float32,[None,784],name='data_x')
y_=tf.placeholder(tf.float32,[None,10],name='data_y')
keep_prob=tf.placeholder(tf.float32,name='keep_prob')#train<1 test =1

hidden1=tf.nn.relu(tf.add(tf.matmul(x,w1),b1),name='h1')
hidden1_drop=tf.nn.dropout(hidden1,keep_prob,name='dropout')
y=tf.nn.softmax(tf.add(tf.matmul(hidden1_drop,w2),b2),name='h2')

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]),name='cross_entropy')
tf.summary.scalar('loss',cross_entropy)
train=tf.train.AdagradOptimizer(lr).minimize(cross_entropy)

merged=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./graph", sess.graph)
    for i in range(3000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        result,_=sess.run([merged,train],feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})
        writer.add_summary(result,i)

    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))
    print("accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

