'''
学习目标：

使用多个特征而非单个特征来进一步提高模型的有效性
调试模型输入数据中的问题
使用测试数据集检查模型是否过拟合验证数据
'''
from __future__ import print_function
import math
from matplotlib import cm,gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.1f}'.format

california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
california_housing_dataframe=california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
#预处理
def preprocess_features(california_housing_dataframe):
    selected_features=california_housing_dataframe[
        ['latitude',
        'longitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income']
    ]
    precessed_features=selected_features.copy()
    #创造合成特征
    precessed_features['rooms_per_person']=(california_housing_dataframe['total_rooms']/california_housing_dataframe['population'])
    return precessed_features
def preprocess_targets(california_housing_dataframe):
    output_targets=pd.DataFrame()
    output_targets['median_house_value']=(california_housing_dataframe['median_house_value']/1000.0)
    return output_targets

'''对于训练集，我们从共 17000 个样本中选择前 12000 个样本。'''
training_examples=preprocess_features(california_housing_dataframe.head(12000))
training_targets=preprocess_targets(california_housing_dataframe.head(12000))
'''对于验证集，我们从共 17000 个样本中选择后 5000 个样本。'''
validation_examples=preprocess_features(california_housing_dataframe.tail(5000))
validation_targets=preprocess_targets(california_housing_dataframe.tail(5000))

'''绘制纬度/经度与房屋价值中位数的曲线图'''
plt.figure(figsize=(13,8))
ax=plt.subplot(1,2,1)
ax.set_title('Validation Data figure1')
ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])
plt.scatter(validation_examples['longitude'],validation_examples['latitude'],
            cmap='coolwarm',c=validation_targets['median_house_value']/validation_targets['median_house_value'].max())
ax=plt.subplot(1,2,2)
ax.set_title('Traing_data figure2')
ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])
plt.scatter(training_examples['longitude'],training_examples['latitude'],
            cmap='coolwarm',c=training_targets['median_house_value']/training_targets['median_house_value'].max())
_=plt.plot()
plt.show()

def my_input_fn(featues,targets,batch_size=1,shuffle=True,num_epochs=None):
    features={key:np.array(value) for key,value in dict(features).items()}



