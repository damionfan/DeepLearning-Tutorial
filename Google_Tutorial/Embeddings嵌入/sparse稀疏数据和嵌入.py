'''稀疏数据和嵌入简介
学习目标：

将影评字符串数据转换为稀疏特征矢量
使用稀疏特征矢量实现情感分析线性模型
通过将数据投射到二维空间的嵌入来实现情感分析 DNN 模型
将嵌入可视化，以便查看模型学到的词语之间的关系
在此练习中，我们将探讨稀疏数据，并使用影评文本数据（来自 ACL 2011 IMDB 数据集）进行嵌入。这些数据已被处理成 tf.Example 格式。
'''
'''https://colab.research.google.com/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb?hl=zh-cn#scrollTo=9MquXy9zLS9B'''
import collections
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
train_url = 'https://dl.google.com/mlcc/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://dl.google.com/mlcc/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

#构建输入管道
#函数来解析训练数据和测试数据（格式为 TFRecord），然后返回一个由特征和相应标签组成的字典。
def _parse_function(record):
    #record: File path to a TFRecord file
    #Returns:
    # A `tuple` `(labels, features)`:
    #   features: A dict of tensors representing the features
    #   labels: A tensor with the corresponding labels.
    features={'terms':tf.VarLenFeature(dtype=tf.string), #terms 是变长的string
              'labels':tf.FixedLenFeature(shape=[1],dtype=tf.float32)}#labels 0/1
    parsed_features=tf.parse_single_example(record,features)
    terms=parsed_features['terms'].values
    labels=parsed_features['labels']
    return {'terms':terms},labels
#训练数据构建一个 TFRecordDataset，并使用上述函数将数据映射到特征和标签
ds=tf.data.TFRecordDataset(train_path)
ds=ds.map(_parse_function)
# print(ds)#<MapDataset shapes: ({terms: (?,)}, (1,)), types: ({terms: tf.string}, tf.float32)>

n=ds.make_one_shot_iterator().get_next()
sess=tf.Session()
sess.run(n)
#建一个正式的输入函数，可以将其传递给 TensorFlow Estimator 对象的 train() 方法。
def _input_fn(input_filenames,num_epochs=None,shuffle=True):
    ds=tf.data.TFRecordDataset(input_filenames)
    ds=ds.map(_parse_function)
    if shuffle:
        ds=ds.shuffle(10000)
    #变长->填充
    ds=ds.padded_batch(25,ds.output_shapes)
    ds=ds.repeat(num_epochs)
    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels
#使用具有稀疏输入和显式词汇表的线性模型
#categorical_column_with_vocabulary_list 函数可使用“字符串-特征矢量”映射来创建特征列。
informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family", "man", "woman", "boy", "girl")
terms_feature_column=tf.feature_column.categorical_column_with_vocabulary_list(key='terms',vocabulary_list=informative_terms)
#embedding column
terms_embedding_column=tf.feature_column.embedding_column(terms_feature_column,dimension=2)

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# feature_columns=[terms_feature_column]
feature_columns=[terms_embedding_column]


classifier = tf.estimator.DNNClassifier(                                      #
  feature_columns=[tf.feature_column.indicator_column(terms_feature_column)], #
  hidden_units=[20,20],                                                       #
  optimizer=my_optimizer,                                                     #
)
# classifier=tf.estimator.LinearClassifier(feature_columns=feature_columns,optimizer=my_optimizer)

classifier.train(input_fn=lambda:_input_fn([train_path]),steps=1000)
evaluation_metrics=classifier.evaluate(input_fn=lambda:_input_fn([train_path]),steps=1000)
print('trianing set metrics')
for m in evaluation_metrics:
    print(m,evaluation_metrics[m])
print('-------------------')

#检查该模型确实在内部使用了嵌入？
classifier.get_variable_names()
#'dnn/input_from_feature_columns/input_layer/terms_embedding/...'
classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape

embedding_matrix=classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

