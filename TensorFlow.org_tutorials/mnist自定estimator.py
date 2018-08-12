import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
'''设置阈值
在tensorflow中有函数可以直接log打印，这个跟ROS系统中打印函数差不多。
TensorFlow使用五个不同级别的日志消息。 
按照上升的顺序，它们是DEBUG，INFO，WARN，ERROR和FATAL。 
当您在任何这些级别配置日志记录时，TensorFlow将输出与该级别相对应的所有日志消息以及所有级别的严重级别。
 例如，如果设置了ERROR的日志记录级别，则会收到包含ERROR和FATAL消息的日志输出，如果设置了一个DEBUG级别，则会从所有五个级别获取日志消息。
 # 默认情况下，TENSFlow在WARN的日志记录级别进行配置，但是在跟踪模型训练时，您需要将级别调整为INFO，这将提供适合操作正在进行的其他反馈。'''


if __name__=='__main__':
    tf.app.run()

def cnn_model_fn(feaures,labels,mode):
    input_layer=tf.reshape(features['x'],[-1,28,28,1])

    conv1=tf.layers.conv2d(inputs=input_layer,filter=32,kernel_size=[5,5],padding="SAME",acitivation=tf.nn.relu)
    pool1=tf.layers.max_pooling_2d(inputs=conv1,pool_size=[2,2],strides=2)

    conv2=tf.layers.conv2d(inputs=pool1,filers=64,kernel_size=[5,5],padding='SAME',activation=tf.nn.relu)
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],stride=2)

    pool2_flat=tf.reshape(pool2,[-1,7*7*64])
    dense=tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)
    dropout=tf.layers.dropout(inputs=dense,rate=0.4,training= (mode==tf.estimator.ModeKeys.TRAIN))

    logits=tf.layers.dense(inputs=dropout,units=10)

    predictions={
        'classes':tf.argmax(logits,1),
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor')
    }

    if mode==tf.estimator.Modekeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    if mode==tf.estimator.Modekeys.TRAIN:
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    eval_metric_ops={
        'accuracy':tf.metrics.accuracy(labels,predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)
