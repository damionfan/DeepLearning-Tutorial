import tensorflow as tf

b=tf.Variable(tf.zeros([100]))
w=tf.Variable(tf.random_uniform([784,100],-1,1))

x=tf.placeholder(name='x')

relu=tf.nn.relu(tf.matmul(x,w)+b)

C=[...]
sess=tf.Session()
for step in range(0,10):
    input=...
    results=sess.run(c,feed_dict={x:input})
    print(step,results)
for i in range(10):
    for d in range(4):
        with tf.device('/gpu:%d'%d):
            pass