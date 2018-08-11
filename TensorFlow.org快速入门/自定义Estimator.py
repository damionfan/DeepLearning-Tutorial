from datetime import date

import tensorflow as tf
import numpy as np
import pandas as pd
#自定Estimator使用的tf.estimator.Estimator的一个实例
#input fn
def train_infput_fn(features,labels,batch_size):
    dataset=tf.data.Dataset.from_tensor_slices(dict(features),labels)
    dataset=dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

#feature column
my_feature_columns=[]
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numric_column(key=key))

#model fn
'''要实现一般的模型函数，您必须执行下列操作：
定义模型。
分别为三种不同模式指定其他计算：
预测
评估
训练'''
classifier=tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns':my_feature_columns,
            'hidden_units':[10,10],
            'n_classes':3
        }
    )

def my_model_fn(features,labels,mode,#mode:tf.estimator.ModeKeys 的实例
                params):
    #features 和 labels 是模型将使用的数据的句柄。mode 参数表示调用程序是请求训练、预测还是评估。
    # 定义模型：一个输入，多个hideen，一个输出
    #定义输入层，把特征字典和feature_columns转换为模型输入
    net=tf.feature_column.input_layer(features,params['feature_columns'])
    #隐藏层
    #layers API 提供一组丰富的函数来定义所有类型的隐藏层，包括卷积层、池化层和丢弃层
    for units in params['hidden_units']:
        net=tf.layers.dense(net,units=units,activation=tf.nn.relu)
    #输出层
    logits=tf.layers.dense(net,params['n_classes'],activation=None)#for tf.nn.softmax


    '''
    train()->	ModeKeys.TRAIN
    evaluate()->	ModeKeys.EVAL
    predict()->	ModeKeys.PREDICT'''
    # classifier = tf.estimator.Estimator(...)
    # classifier.train(input_fn=lambda: my_input_fn(FILE_TRAIN, True, 500))
    '''模型函数必须提供代码来处理全部三个 mode 值。对于每个 mode 值，
    您的代码都必须返回 tf.estimator.EstimatorSpec 的一个实例，其中包含调用程序需要的信息。我们来详细了解各个 mode。'''
    # 预测
    predicted_classes = tf.argmax(logits, 1)#这里我猜想是一个样本一个数据，多个合成predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)







