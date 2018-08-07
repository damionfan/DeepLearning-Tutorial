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
    #targets:pd.Series
    features={key:np.array(value) for key,value in dict(features).items()}
    ds=Dataset.from_tensor_slices((features,targets))#这两个都是dict
    ds=ds.batch(batch_size).repeat(num_epochs)
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
    linear_regressor=tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples),
                                                 optimizer=my_optimizer)
    #这里构造的特征列的参数key：是DataFrame!!!

    #input fn
    training_input_fn=lambda:my_input_fn(training_examples,training_targets['median_house_value'],batch_size,)#targets这里是一个pd.Series
    predict_training_input_fn=lambda:my_input_fn(training_examples,training_targets['median_house_value'],num_epochs=1,shuffle=False)
    predict_validation_input_fn=lambda:my_input_fn(validation_examples,validation_targets['median_house_value'],num_epochs=1,shuffle=False)

    print('train and record rmse')
    training_rmse=[]
    validation_rmse=[]
    for period in range(0,periods):
        #train
        linear_regressor.train(input_fn=training_input_fn,steps=steps_per_period)
        #validation and predict
        training_predictions=linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions=np.array([item['predictions'][0] for item in  training_predictions])#这里还是不太懂 这里好像是默认就是'predictions'

        validation_predictions=linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions=np.array([item['predictions'][0] for item in validation_predictions])

        #loss    这里有没有列名都行
        training_root_mean_squared_error=math.sqrt(metrics.mean_squared_error(training_predictions,training_targets['median_house_value']))
        validation_root_mean_squared_error=math.sqrt(metrics.mean_squared_error(validation_predictions,validation_targets['median_house_value']))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print('train done')

    #plot
    plt.figure(1)
    plt.ylabel('RMSE')
    plt.xlabel('Reriods')
    plt.title('RMSE VS PERIODS figure1')
    plt.tight_layout()
    plt.plot(training_rmse,label='training_rmse')
    plt.plot(validation_rmse,label='validation_rmse')
    plt.legend()
    # plt.show()
    return linear_regressor

minimal_features=['median_income','latitude']
minimal_training_examples=training_examples[minimal_features]
minimal_validation_examples=validation_examples[minimal_features]

# train_model(learning_rate=1e-2,steps=500,batch_size=5,
#             training_examples=minimal_training_examples,
#             training_targets=training_targets,
#             validation_examples=minimal_validation_examples,
#             validation_targets=validation_targets)

plt.scatter(training_examples['latitude'],training_targets['median_house_value'])
# 对纬度进行分箱。在 Pandas 中使用 Series.apply  ->Boolean类型
LATITUDE_RANGES=zip(range(32,44),range(33,45))
def select_and_transform_features(source_df):
    select_examples=pd.DataFrame()
    select_examples['median_income']=source_df['median_income']
    for r in LATITUDE_RANGES:
        select_examples['latitude_%d_to_%d'%r]=source_df['latitude'].apply(
            lambda l:1.0 if l>=r[0] and l<r[1] else 0.0
        )#lambda x: v1 if True else v2
    return select_examples
selected_training_examples = select_and_transform_features(training_examples)
selected_validation_examples = select_and_transform_features(validation_examples)
train_model(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=selected_training_examples,
    training_targets=training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation_targets)