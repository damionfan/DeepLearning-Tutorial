import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# a=tf.constant(2,name='a')
# b=tf.constant(3,name='b')
# x=tf.add(a,b,name='add')
#
# with tf.Session() as sess:
#     #add this line to use TensorBoard
#     writer=tf.summary.FileWriter('./graphs',sess.graph)
#     print(sess.run(x))
# writer.close()#close the writer!!! dont frogot it !

# '''
# tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)
# '''
# a=tf.constant(2,shape=[2,2])
# tf.InteractiveSession()
# a.eval()
# #[[2,2],[2,2]]
# b=tf.constant([2,1],shape=[3,3])
# b.eval()
# #[[2,1,1],
# # [1,1,1],
# # [1,1,1]]

a=tf.constant([2,2],name='a')
b=tf.constant([[0,1],[2,3]],name='b')
x=tf.add(a,b,name='add')
y=tf.multiply(a,b,name='mul')
with tf.Session() as sess:
    sess.run([x,y])
    print(x,y)