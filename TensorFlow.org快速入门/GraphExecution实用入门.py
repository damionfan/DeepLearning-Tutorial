import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.tensor_forest.python.ops.gen_model_ops import feature_usage_counts

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
CSV_COLUMN_NAMES=['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
lr=1e-3
#load data
def load_data(label_name='Species'):

    train_path=tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],origin=TRAIN_URL)
    '''
        Args:
            fname=下载的路径 默认下载到/为 ：~/.keras/datasets/xxxx.xxx
            origin=原始的url
        Return:
            下载文件的路径
    '''
    train=pd.read_csv(train_path,names=CSV_COLUMN_NAMES,#names:列名
                      header=0 #从数据开始，等于忽略原来列名，改成新的列名
                        )
    train_features,train_label=train,train.pop(label_name)#just a another name

    test_path=tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)
    test=pd.read_csv(test_path,names=CSV_COLUMN_NAMES,header=0)
    test_features,test_label=test,test.pop(label_name)

    #return DataFrame
    return (train_features,train_label),(test_features,test_label)


#get data
(train_feature,train_label),(test_feature,test_label)=load_data()


#feature column
my_feature_columns=[]
for key in dict(train_feature).keys():
    # print(key)
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


#input_fn
def train_input_fn(features,labels,batch_size):
    dataset=tf.data.Dataset.from_tensor_slices((dict(features),labels))
    dataset=dataset.shuffle(buffer_size=10000).repeat(count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


#classifier
optimizer=tf.train.AdamOptimizer(learning_rate=lr)
optimizer=tf.contrib.estimator.clip_gradients_by_norm(optimizer,5.0)
classifier=tf.estimator.DNNClassifier(hidden_units=[10,10],feature_columns=my_feature_columns,n_classes=3,optimizer=optimizer)

#train
classifier.train(
    input_fn=lambda:train_input_fn(train_feature,train_label,args.batch_size),#这个args.batch_size
    steps=args.train_steps
)

#evaluate
# lassifier.evaluate 必须从测试集（而非训练集）中获取样本。
# 换言之，为了公正地评估模型的效果，用于评估模型的样本一定不能与用于训练模型的样本相同
def eval_input_fn(features,labels=None,batch_size=None):

    if labels is None:
        inputs=features
    else:
        inputs=(features,labels)

    dataset=tf.data.Dataset.from_tensor.slices(inputs)

    #batch
    '''在开发一个程序时候，与其让它运行时崩溃，不如在它出现错误条件时就崩溃（返回错误）。这时候断言assert 就显得非常有用。

assert的语法格式：

assert expression
1
它的等价语句为：

if not expression:
    raise AssertionError'''
    assert batch_size is not None ,'batch_size must not be None'
    dataset=dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

eval_result=classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_feature,test_label,arg.batch_size)
)

#predict
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x,
                                  labels=None,
                                  batch_size=args.batch_size))
'''predict 方法返回一个 Python 可迭代对象，为每个样本生成一个预测结果字典。此字典包含几个键。
probabilities 键存储的是一个由三个浮点值组成的列表，每个浮点值表示输入样本是特定鸢尾花品种的概率
class_ids 键存储的是一个 1 元素数组，用于标识可能性最大的品种'''
for pred_dict,expec in zip(predictions,expected):
    template=('\nPrediction is "{}" ({:.1f}%),expected"{}" ')
    class_id=pred_dict['class_ids'][0]
    probability=pred_dict['probabilities'][class_id]
    print(template.format(iris_data_SPECIES[class_id]),100*probability,expec)




