import tensorflow as tf
#
# a = tf.add(2,3)
#
# # sess=tf.Session()
# # print(sess.run(a))
# # sess.close()
# with tf.Session() as sess:#auto close!
#     print(sess.run(a))
# x=2;y=3
# op1=tf.add(x,y)
# op2=tf.multiply(x,y)
# useless=tf.add(op1,x)
# op3=tf.pow(op2,op1)
# with tf.Session() as sess:
#     op3=sess.run(op3)
# print(op3)
# print(useless)


#
# #to add operators to a graph ,set it as default
# g=tf.Graph()
# with g.as_default():
#     x=tf.add(3,5)
#
# sess=tf.Session(graph=g)
# sess.run(x)
# sess.close()
#
# # with tf.Session() as sess:
# #     sess.run(x)
#
# # #to handle the default graph
# # g=tf.get_default_graph()


# #Do not mix default graph and user created graphs
# g=tf.Graph()
# #add ops to the default graph
# a=tf.constant(3)
# #add ops to the user created graph
# with g.as_default():
#     b=tf.constant(5)



# #Do not mix default graph and user created graphs
# g1=tf.get_default_graph()
# g2=tf.Graph()
# #add ops to the default graph
# with g1.as_default():
#     a=tf.constant(2)
# #add ops to user created graph
# with g2.as_default():
#     b=tf.constant(5)



