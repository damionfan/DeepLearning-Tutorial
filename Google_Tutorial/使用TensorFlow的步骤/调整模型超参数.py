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


def train_model(learning_rate,steps,batch_size,input_feature='total_rooms'):
    '''使用一个feature去train线性模型
    Args:
        lr: float
        step:  非零 训练的总次数，一次一更新
        batch_size:非零 一次更新需要的量
        input_feature:'string',California_housing_dataframe中特定的一行作为特征列
    '''
    periods=10#要多少次存储数据
    steps_per_periods=steps/periods

    my_feature=input_feature
    my_feature_data=california_housing_dataframe[[my_feature]]#两个[[]]有个列名。。。
    my_label='median_house_value'
    targets=california_housing_dataframe[my_label]

    #特征列
    feature_columns=[tf.feature_columns.numeric_column(my_feature)]
    #创造 linear regressor 对象
    my_optimizer=
