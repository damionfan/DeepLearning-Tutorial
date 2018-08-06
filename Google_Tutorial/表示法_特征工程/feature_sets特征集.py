from __future__ import print_function

import math
from matplotlib import cm,gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

from Google_Tutorial.使用TensorFlow的步骤.TfExample import my_feature

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.1f}'.format

california_housing_dataframe=pd.read_csv('california_housing_train.csv',sep=',')
california_housing_dataframe=california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
    select_fetures=california_housing_dataframe[
        [
            'latitude',
            'longitude',
            'housing_median_age',
            'total_rooms',
            'total_bedrooms',
            'population',
            'households',
            'median_income'
        ]
    ]
    precessed_features=select_fetures.copy()
    precessed_features['rooms_per_person']=(
        california_housing_dataframe['total_rooms']/
        california_housing_dataframe['population']
    )
    return precessed_features
def preprocess_targets(california_housing_dataframe):
    output_targets=pd.DataFrame()
    output_targets['median_house_value']=california_housing_dataframe['median_house_value']/1000.0
    return output_targets

#12000 for train
training_examples=preprocess_features(california_housing_dataframe.head(12000))
training_targets=preprocess_targets(california_housing_dataframe.head(12000))

#5000 for validation
validation_examples=preprocess_features(california_housing_dataframe.tail(5000))
validation_targets=preprocess_targets(california_housing_dataframe.tail(5000))

#检查两遍 确保万无一失
print('train calibration')
print(training_examples.describe())
print(training_targets.describe())

print('validation calibration')
print(validation_examples.describe())
print(validation_targets.describe())

print('-------------------------------------')

'''相关性值具有以下含义：

-1.0：完全负相关
0.0：不相关
1.0：完全正相关'''

correlation_dataframe=training_examples.copy()
correlation_dataframe['target']=training_targets['median_house_value']
correlation_dataframe.corr()#计算相关性，排除null/NA 默认皮尔森
#输出是一个表格形式 n*n

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

def my_input_fn(features,targets,batch_size=1,shuffle=False,num_epochs=None):
    features={key:np.array(value) for key,value in dict(features).items()}
    ds=Dataset.from_tensor_slices((features,targets))
    ds=df.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds=ds.shuffle(10000)
    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels

def train_model(learning_rate,steps,batch_size,training_examples,training_targets,
                validation_examples,validation_targets):
    periods=10
    steps_per_period=steps/periods

    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
    liear_regressor=tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples),
                                                 optimizer=my_optimizer)
    #这里构造的特征列的参数key：是DataFrame!!!


