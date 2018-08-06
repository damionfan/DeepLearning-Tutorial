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
pd.options.display.float_format='{:.1f}'.format#接收浮点返回指定格式

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
my_feature= california_housing_dataframe[['total_rooms']]
# print(my_features)
'''numeric_column 定义特征列，这样会将其数据指定为数值'''
feature_columns=[tf.feature_column.numeric_column('total_rooms')]#key:标识输入要素的唯一字符串

'''--------------step2:定义目标------------------------------------------'''
#define label
targets=california_housing_dataframe['median_house_value']

'''--------------step3:配置LinearRegressor----------------------------------'''
#通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
#GD
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-7)
#在使用梯度之前切分梯度（optimizer,clip_norm) 裁剪，确保数值稳定性以及防止梯度爆炸。

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
    #dataset使用的话用iterator
    ds=Dataset.from_tensor_slices((features,targets))#warning:2G limit 如果有numpy则作为constant
    ds=ds.batch(batch_size).repeat(num_epochs)

    #shuffle
    if shuffle:
        ds=ds.shuffle(buffer_size=10000)

    #返回下一个batch
    #terator = dataset.make_one_shot_iterator()从dataset中实例化了一个Iterator，这个Iterator是一个“one shot iterator”，即只能从头到尾读取一次
    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels
'''question：features，targets 都是pandas DataFrame，第一个是dict，第二个是pandas DataFrame那么对features和labels有什么格式要求吗？
    第一个其实把DataFrame改成了一个dict,targets也是一个dict,从理解上说的话'''

'''---------------------step5:训练模型------------------------------------------'''
_=linear_regressor.train(input_fn=lambda:my_input_fn(my_feature,targets),steps=100)

'''----------------------step6:评估模型------------------------------------------'''
# 训练误差可以衡量您的模型与训练数据的拟合情况，但并不能衡量模型泛化到新数据的效果。
# 在后面的练习中，您将探索如何拆分数据以评估模型的泛化能力。
# 创造一个input_fn为了prediction
# 这是为了做预测，所以不需要shuffle和repeat
prediction_input_fn=lambda:my_input_fn(my_feature,targets,num_epochs=1,shuffle=False)

#返回一个Estimator.predict 对象 是含有预测值的tensor
predictions=linear_regressor.predict(input_fn=prediction_input_fn)

# print('---------------------')
# print(predictions) <generator object Estimator.predict at 0x00000133BBABD7D8>
#把predictions->numpy array，我们进行误差测量
'''question:????'''
predictions=np.array([item['predictions'][0] for item in predictions])
print('--------------------')
print(predictions)
print('----------------------')
#输出均方误差 和均方根误差
mean_squared_error= metrics.mean_squared_error(predictions,targets)
root_mean_squared_error=math.sqrt(mean_squared_error)
print('均方误差  ：%0.3f'%mean_squared_error)
print('均方根误差：%0.3f'%root_mean_squared_error)

min_house_value=california_housing_dataframe['median_house_value'].min()
max_house_value=california_housing_dataframe['median_house_value'].max()
min_max_difference=max_house_value-min_house_value

print("min:%0.3f"%min_house_value)
print("man:%0.3f"%max_house_value)
print("dif:%0.3f"%min_max_difference)

calibration_data=pd.DataFrame()
calibration_data['predictions']=pd.Series(predictions)
calibration_data['targets']=pd.Series(targets)
print(calibration_data.describe())
'''---------------plot------------'''
sample=california_housing_dataframe.sample(n=300)#取样可视化
x_0=sample['total_rooms'].min()
x_1=sample['total_rooms'].max()
#取回最后的weights和bias
weight=linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias=linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0=weight*x_0+bias
y_1=weight*x_1+bias
#plot
plt.plot([x_0,x_1],[y_0,y_1],c='r')
#labels
plt.ylabel('median_house_value')
plt.xlabel('total_rooms')

#plot sample
plt.scatter(sample['total_rooms'],sample['median_house_value'])

#display
plt.show()

