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

def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
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
    feature_columns=[tf.feature_column.numeric_column(my_feature)]

    #input 函数
    training_input_fn=lambda:my_input_fn(my_feature_data,targets,batch_size=batch_size)
    prediction_input_fn=lambda:my_input_fn(my_feature_data,targets,shuffle=False,num_epochs=1)

    #线性回归对象
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
    linear_regressor=tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)

    #plot
    plt.figure(figsize=(15,6))#使用figsize 与 dpi 参数能够设置图表尺寸与DPI，
    plt.subplot(1,2,1)
    plt.title('learned line by period')
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample=california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature],sample[my_label])
    colors=[cm.coolwarm(x) for x in np.linspace(-1,1,periods)]

    #训练model 定期评估
    print('训练model')
    print('RMSE(在训练数据）')
    root_mean_squared_errors=[]
    for period in range(0,periods):
        #train
        linear_regressor.train(input_fn=training_input_fn,steps=steps_per_periods)
        #进行预测 在每stpes/periods之后
        predictions=linear_regressor.predict(input_fn=prediction_input_fn)
        predictions=np.array([item['predictions'][0] for item in predictions])#!!!!!!!!!!
        #计算loss
        root_mean_squared_error=math.sqrt(metrics.mean_squared_error(predictions,targets))

        print('period %02d:%0.2f'%(period,root_mean_squared_error))

        root_mean_squared_errors.append(root_mean_squared_errors)

        #找到weights,biases
        y_extents=np.array([0,sample[my_label].max()])

        weight=linear_regressor.get_variable_value('linear/linear_model/%s/weights'%input_feature)[0]
        bias=linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents=(y_extents-bias)/weight
        x_extents=np.maximum(np.minimum(x_extents,sample[my_feature].max()),sample[my_feature].min())#!!!!
        y_extents=weight*x_extents+bias

        plt.plot(x_extents,y_extents,color=colors[period])
    print('model training finished')

    #plot loss
    plt.subplot(1,2,2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('RMSE vs PERIODS')
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    #输出输入数据
    calibration_data=pd.DataFrame()
    calibration_data['predictions']=pd.Series(predictions)
    calibration_data['targets']=pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

train_model(learning_rate=0.0001,steps=100,batch_size=1)


