import time
import numpy as np
import tensorflow as tf
import reader

class PTBInput(object):
    def __init__(self,config,data,name=None):
        self.batch_size=batch_size=config.batch_size
        self.num_steps=num_steps=config.num_steps
        self.epoch_size=((len(data)//batchsize)-1)// num_steps #//浮点除法，结果四舍五入
        self.input_data,self.targets=reader.ptb_producer(data,batch_size,num_steps,name=name)
class PTBModel(object):
    def __init__(self,is_training,config,input_):
        self._input=input_
        batch_size=input_.batch_size
        num_steps=input_.num_steps
        size=config.hidden_size
        vocab_size=config.vocab_size
'''tf.contrib.rnn.BasicLSTMCell lstm_cell之后接一个Dropout,这里使用tf.contrib.rnn.DropoutWrapper行数，最后使用RNN堆叠函数tf.contrib.rnn.MultiRNNCell把前面的lstm_cell多层堆叠到cell'''
def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(size,forget_bias=0,state_is_tuple=True)

attn_cell=lstm_cell
if is_training and config.keep_prob<1:
    def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(),output_keep_prob=config.keep_prob
        )
    cell=tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)],state_is_tuple_True
    )
    self._initial