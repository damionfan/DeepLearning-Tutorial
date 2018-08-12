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
    您的代码都必须返回 tf.estimator.EstimatorSpec 的一个实例，其中包含预测信息。我们来详细了解各个 mode。'''
    # 预测
    '''question: 这里的返回的值是几个,？还有class_ids是怎么样的,因为有个batch，所以我觉得可能是多维的'''
    predicted_classes = tf.argmax(logits, 1)#这里我猜想是一个样本一个数据，多个合成predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    #loss train和evaluate都需要计算loss，这个是优化的目标
    #tf.losses.sparse_softmax_cross_entropy 返回batch的平均 sparse:稀疏
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)


    #指标的话可以使用tf.metircs
    accuracy=tf.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')
    #returns:
    # accuracy: total/count
    # update_op:更新total,count，和acc match

    metrics={
        'accuracy':accuracy
    }
    tf.summary.scalar('accuracy',accuracy[1])
    '''accuracy：准确率由下列两行记录：

eval_metric_ops={'my_accuracy': accuracy}（评估期间）。
tf.summary.scalar('accuracy', accuracy[1])（训练期间）。
这些 Tensorboard 图是务必要将 global_step 传递给优化器的 minimize 方法的主要原因之一。如果没有它，模型就无法记录这些图的 x 坐标。

注意 my_accuracy 和 loss 图中的以下内容：

橙线表示训练。
蓝点表示评估。
在训练期间，系统会随着批次的处理定期记录摘要信息（橙线），因此它会变成一个跨越 x 轴范围的图形。

相比之下，评估在每次调用 evaluate 时仅在图上生成一个点。此点包含整个评估调用的平均值。它在图上没有宽度，因为它完全根据特定训练步（一个检查点）的模型状态进行评估。'''
    #tf.summary.scalar 会在 TRAIN 和 EVAL 模式下向 TensorBoard 提供准确率（后文将对此进行详细的介绍）
    #evaluate model=ModeKeys.EVAL
    # returns:包含loss和多个可选的指标的tf.estimator.EstimatorSpec

    if mode==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,loss,eval_metric_ops=metrics
        )

    #mode=ModeKeys.TRAIN
    #return loss,train_op 的EstimatorSpec

    optimizer=tf.train.AdagradOptimizer(learning_rate=0.1)
    #global_step 对于 TensorBoard 图能否正常运行至关重要。
    # 只需调用 tf.train.get_global_step 并将结果传递给 minimize 的 global_step 参数即可。
    train_op=optimizer.minimize(loss,global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)







