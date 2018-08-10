from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)#logging_hook

#our application logic wil be added here
# if __name__=='__main__':#从这文件执行 而不是导入（import）的话 会run()
#     tf.app.run()

def cnn_model_fn(features,labels,mode):
    """Model function for CNN"""
    #input layer
    input_layer=tf.reshape(features['x'],[-1,28,28,1])#Reshape X to 4-D tensor: [batch_size, width, height, channels] -1自动计算batch_size

    #Convolutional layer #1
    conv1=tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )#[batch_size,28,28,32]
    #Pooling layer #1
    pool1=tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        stride=2,
        )#[batch_size,14,14,32]

    #Convolutional layer #2 and pooling layer #2
    conv2=tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )#[batch_size,14,14,64]!!64-channel
    pool2=tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
    )#[batch_size,7,7,64]

    #Dense layer
    '''
Logits是一个超载的术语，它可能意味着许多不同的东西：

在数学中，Logit是一种将概率（[0, 1]）映射到R（(-inf, inf)）的函数，

在这里输入图像描述

0.5的概率对应于0的对数。负对数的概率小于0.5，正值> 0.5。

在ML中，它可以

分类模型生成的原始（非归一化）预测的向量，通常将其传递给归一化函数。如果模型正在解决多类分类问题，则logits通常会成为softmax函数的输入。然后softmax函数为每个可能的类生成一个（标准化）概率向量，其中一个值为一个值。

Logit 有时也会引用sigmoid函数的元素反向。

Logit是一种将概率映射[0, 1]到的函数[-inf, +inf]。

SOFTMAX是映射功能[-inf, +inf]，以[0, 1]类似S形。但是Softmax也将这些值的总和（输出矢量）归一化为1。

Tensorflow“with logit”：这意味着您正在应用softmax函数来对数字进行logit归一化处理。input_vector / logit未规范化，可以从[-inf，inf]进行缩放。

    '''
    pool2_flat=tf.reshape(pool2,[-1,7*7*64])#flatted
    dense=tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout=tf.layers.dropout(#是对tf.nn.dropout 的包装 rate=1-keep_prob
        inputs=dense,
        rate=0.4,#丢包率
        training=mode ==tf.estimator.ModeKeys.TRAIN #Whether to return the output in training mode (apply dropout) or in inference mode (return the input untouched).
    )                                               #看是不是在Train模式，是的话才使用dropout

    #Logits layer 神经网络中的最后一层是逻辑层，它将返回我们预测的原始值 作为softmax的输入层
    logits=tf.layers.dense(inputs=dropout,units=10)




    predictions={
        #Generate predictions (for Predict and eval mode)
        'classes':tf.argmax(input=logits,axis=1),
        #Add 'softmax_tensor' to the graph,It is used for Predict and by the 'logging_hook'(每N个本地步骤打印给定张量，每N秒打印一次，或结束打印。)
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor')
    }

    if mode==tf.estimator.ModeKeys.PREDICT:#?
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    #Calculate Loss (for both Train and eval modes)
    onehot_labels=tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)#变成ont-hot encoding
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    #configure the Training op (for train mode)
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimzer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())#global_step++
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    #Add evaluation metrics(for eval mode)
    eval_metric_ops={
        "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)#?


def main(unused_argv):
    #load training and eval data
    mnist=tf.contrib.learn.datasets.load_dataset('mnist')
    train_data=mnist.train.images#returns np array
    train_labels=np.asarray(mnist.train.labels,dtype=np.int32)
    eval_data=mnist.test.images
    eval_labels=np.asarray(mnist.test.labels,dtype=np.int32)

    #Create the Estimator
    mnist_classifier=tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='/tmp/mnist_convent_model')#模型数据保存位置

    #set up logging for predictions
    tensor_to_log={'probabilities':'softmax_tensor'}
    logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)

    #Train the model
    train_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )
    #evaluate tehe model and print results
    eval_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x',eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results=mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_result)
