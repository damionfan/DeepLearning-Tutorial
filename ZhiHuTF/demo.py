import tensorflow as tf

hello=tf.constant('hello')
with tf.Session() as sess:
   print(hello.eval())
#字符串之前的b不是字符串的一部分，而是表示它存储为bytes。 在Python 3.x中，bytes和string对象是不同的（即使你可以很容易地从一个转换到另一个）。
#
#所以，b'hello'表示字符串hello被表示为字节; 这与您输入的字符串相同。