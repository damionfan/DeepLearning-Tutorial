import tensorflow as tf
"""
tf.gradients(y,[xs])
take derivative of y with respect to each tensor in the list [xs]
"""
# x=tf.Variable(2.0)
# y=2.0*(x**3)
# z=3.0+y**2
# grad_z=tf.gradients(z,[x,y])
# with tf.Session() as sess:
#     sess.run(x.initializer)
#     print(sess.run(grad_z))#[768.0,32.0]
'''
object oriented programming 面向对象编程
'''
class SkipGramModel:
    '''bulid the graph gor word2vec model'''
    def __init__(self,params):
        pass

    def _create_placeholder(self):
        '''step1:define the placeholders for input and output'''
        pass

    def _create_embedding(self):
        '''step2: define weights,in word2vec,it's actually the weights that we care about '''
        pass

    def _create_loss(self):
        '''step3+4:define the inference + the loss function '''
        pass

    def _create_optimizer(self):
        '''step5:define optimizer'''
        pass
