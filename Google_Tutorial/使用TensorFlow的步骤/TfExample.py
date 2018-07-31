import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import cm,gridspec
from sklearn import metrics
from tensorflow.python.data import Dataset


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.1f}'.format

california_housing_dataframe=pd.read_csv('california_housing_train.csv',sep=',')
'''1.随机化处理2.把median_house_value调整为以千为单位'''
california_housing_dataframe=california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value']/=1000.0
# print(california_housing_dataframe.head())
'''----------------尝试预测 median_house_value----------------'''
'''我们将使用 TensorFlow Estimator API 提供的 LinearRegressor 接口。此 API 负责处理大量低级别模型搭建工作，并会提供执行模型训练、评估和推理方法。'''

'''--------------step1:定义特征并配置特征列---------------------------'''
'''特征列宝石调整额数据类型，进有特征数据的表述，不含数据本身，好像是dtype'''
#定义input feature ：total_rooms
my_features= california_housing_dataframe[['total_rooms']]
# print(my_features)
'''numeric_column 定义特征列，这样会将其数据指定为数值'''
feature_columns=[tf.feature_column.numeric_column('total_rooms')]

'''--------------step2:定义目标------------------------------------------'''
#define label
targets=california_housing_dataframe['median_house_value']

'''--------------step3:配置LinearRegressor----------------------------------'''
#通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
#GD
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-7)
my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)

#配置
linear_regressor=tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)

'''----------------step4:定义输入函数-----------------------------------'''
# 我们将 Pandas 特征数据转换成 NumPy 数组字典。
# 然后，我们可以使用 TensorFlow Dataset API 根据我们的数据构建 Dataset 对象，
# 并将数据拆分成大小为 batch_size 的多批数据，以按照指定周期数 (num_epochs) 进行重复。
# 默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。
# 如果 shuffle 设置为 True，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。
# buffer_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。
# 最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。

def my_input_fn(features,targets,batch_size=1,shuffle=Ture,num_epoches=None):
    '''返回一个线性回归的模型
    Args:
        features:pandas DataFrame
        targets:pandas DataFrame
        batch_size=size of batches to be passed to the model
        shuffle:whether to shuffle the data
        num_epochs:number of epochs for which data should be repeated ,None=indefinitely
    Returns:
        Tuple of (features,labels) for next data batch
    '''
    #pandas data -> numpy array
    features={key:np.array(value) for key,value in dict(features).items()}#每一列 列名：数据 value是列表！
    #构造数据集dataset，配置batching/repeating
    ds=Dataset.from_tensor_slices((features,targets))#warning:2G limit
    ds=ds.batch(batch_size).repeat(num_epochs)

    #shuffle
    if shuffle:
        ds=ds.shuffle(buffer_size=10000)

    #返回下一个batch
    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels
'''question：features，targets 都是pandas DataFrame，第一个是dict，第二个是pandas DataFrame那么对features和labels有什么格式要求吗？'''