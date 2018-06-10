import tensorflow as tf
hello=tf.constant("l love you",dtype=tf.string)
with tf.Session() as sess:
   print(sess.run(hello))