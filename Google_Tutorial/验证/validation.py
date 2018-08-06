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
plt.plot()

def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
    #pd.DataFrame->dict(np.array)
    features={key:np.array(value) for key,value in dict(features).items()}
    #dataset
    ds=Dataset.from_tensor_slices((features,targets))
    ds=ds.batch(batch_size).repeat(num_epochs)

    #shuffle
    if shuffle:
        ds=ds.shuffle(10000)

    #Return next batch
    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels

#特征列
def construct_feature_column(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])#set 无序不重复

def train_model(learning_rate,steps,batch_size,training_examples,training_targets,validation_examples,validation_targets):
    periods=2
    steps_per_period=steps/periods

    #optimizer
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)

    linear_regressor=tf.estimator.LinearRegressor(feature_columns=construct_feature_column(training_examples),optimizer=my_optimizer)

    #input func
    training_input_fn=lambda:my_input_fn(training_examples,training_targets['median_house_value'],batch_size=batch_size)
    predict_training_input_fn=lambda:my_input_fn(training_examples,training_targets['median_house_value'],shuffle=False,num_epochs=1)#就是用训练数据predict
    predict_validation_input_fn=lambda:my_input_fn(validation_examples,validation_targets['median_house_value'],shuffle=False,num_epochs=1)#??

    #Train
    print('train')
    training_rmse=[]
    validation_rmse=[]
    for period in range(0,periods):
        #train
        linear_regressor.train(input_fn=training_input_fn,steps=steps_per_period)
        #predict
        training_predictions=linear_regressor.predict(predict_training_input_fn)
        training_predictions=np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions=linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions=np.array([item['predictions'][0] for item in validation_predictions])
        #loss
        training_root_mean_squared_error=math.sqrt(metrics.mean_squared_error(training_predictions,training_targets))
        validation_root_mean_squared_error=math.sqrt(metrics.mean_squared_error(validation_predictions,validation_targets))

        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print('train done')

    #plot
    plt.figure(2)
    plt.ylabel('RMSE')
    plt.xlabel('periods')
    plt.title('RMSE periods figure3')
    plt.tight_layout()#自动调整子图参数
    plt.plot(training_rmse,label='training_rmse')
    plt.plot(validation_rmse,label='validation_rmse')
    plt.legend()#图例
    plt.show()
    return linear_regressor

linear_regressor=train_model(
    learning_rate=3e-5,
    steps=50,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)



