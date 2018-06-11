import tensorflow as tf

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
c=tf.add(a,b)
d=tf.multiply(a,b)
with tf.Session() as sess:
    print(sess.run(c,feed_dict={a:2,b:3}))
    print(sess.run(d,feed_dict={a:2,b:3}))