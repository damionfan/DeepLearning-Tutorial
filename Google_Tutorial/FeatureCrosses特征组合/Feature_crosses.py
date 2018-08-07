from __future__ import print_function
import math
from matplotlib import cm,gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.1f}'.format

california_housing_dataframe=pd.read_csv('california_housing_train.csv',sep=',')
california_housing_dataframe=california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_frame):
    selected_features=california_housing_dataframe[
        [
            'latitude',
            'longitude',
            'housing_median_age',
            'total_rooms',
            'population',
            'households',
            'median_income'
        ]
    ]
    processed_features=selected_features.copy()#important
    processed_features['rooms_per_person']=california_housing_dataframe['total_rooms']/california_housing_dataframe['population']
    return processed_features
def preprocess_targets(california_housing_frame):
    output_targts=pd.DataFrame()
    output_targts['median_house_value']=california_housing_dataframe['median_house_value']/1000.0
    return output_targts

training_examples=preprocess_features(california_housing_dataframe.head(12000))
training_targets=preprocess_targets(california_housing_dataframe.head(12000))
validation_examples=preprocess_features(california_housing_dataframe.tail(5000))
validation_targets=preprocess_targets(california_housing_dataframe.tail(5000))

#check

def construct_feature_columns(input_feaures):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_feaures])

def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
    features={key:np.array(value)
              for key,value in dict(features).items()}#important
    ds=Dataset.from_tensor_slices((features,targets))
    ds=ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds=ds.shuffle(10000)
    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels


'''高维度线性模型可受益于使用一种基于梯度的优化方法，叫做 FTRL。
该算法的优势是针对不同系数以不同方式调整学习速率，如果某些特征很少采用非零值，
该算法可能比较实用（也非常适合支持 L1 正则化）。我们可以使用 FtrlOptimizer 来应用 FTRL。'''

def train_model(learning_rate,steps,batch_size,feature_columns,training_examples,training_targets,validation_examples,validation_targets):
    periods=10
    steps_per_period=steps/periods
    #LinearRegressor
    my_optimizer=tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
    linear_regressor=tf.estimator.LinearRegressor(feature_columns,optimizer=my_optimizer)

    training_input_fn=lambda:my_input_fn(training_examples,training_targets['median_house_value'],batch_size)
    predict_training_input_fn=lambda:my_input_fn(training_examples,training_targets['median_house_value'],num_epochs=1,shuffle=False)
    predict_validation_input_fn=lambda:my_input_fn(validation_examples,validation_targets['median_house_value'],num_epochs=1,shuffle=False)

    training_rmse=[]
    validation_rmse=[]
    for period in range(0,periods):
        #train
        linear_regressor.train(input_fn=training_input_fn,steps=steps_per_period)
        #predict
        training_predictions=linear_regressor.predict(input_fn=predict_training_input_fn)
        validation_predictions=linear_regressor.predict(input_fn=predict_validation_input_fn)
        '''！！！！！！！！！！！！！！predictions类型不知道 元素是array'''
        training_predictions=np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions=np.array([item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error=math.sqrt(metrics.mean_squared_error(training_predictions,training_targets))
        validation_root_mean_squared_error=math.sqrt(metrics.mean_squared_error(validation_predictions,validation_targets))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print('train done')

    #plot
    plt.ylabel('RMSE')
    plt.xlabel("Reriods")
    plt.title('figure1')
    plt.tight_layout()
    plt.plot(training_rmse,label='training_rmse')
    plt.plot(validation_rmse,label='validation_rmse')
    plt.legend()

    return linear_regressor
train_model(
    learning_rate=1.0,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(training_examples),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

plt.show()

'''分桶特征定义特征列，我们可以使用 bucketized_column（而不是使用 numeric_column），
该列将数字列作为输入，并使用 boundardies 参数中指定的分桶边界将其转换为分桶特征'''
def get_quantile_based_boudaries(feature_values,num_buckets):
    boundaries=np.arange(1.0,num_buckets) / num_buckets#1/n,2/n,,,,n-1/n
    quantiles=feature_values.quantile(boundaries)#返回一个DataFrame，值是分位数
    return [quantiles[q] for q in quantiles.keys()]
#把household分成7分 7 buckets
households=tf.feature_column.numeric_column('households')
bucketized_households=tf.feature_column.bucketized_column(households,boundaries=get_quantile_based_boudaries(california_housing_dataframe['households'],7))

#longitude ->10 buckets
longitude=tf.feature_column.numeric_column('longitude')
bucketized_longitude=tf.feature_column.bucketized_column(longitude,boundaries=get_quantile_based_boudaries(california_housing_dataframe['longitude'],10))
#分桶
'''使用分桶特征列训练模型
def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """ 
  households = tf.feature_column.numeric_column("households")
  longitude = tf.feature_column.numeric_column("longitude")
  latitude = tf.feature_column.numeric_column("latitude")
  housing_median_age = tf.feature_column.numeric_column("housing_median_age")
  median_income = tf.feature_column.numeric_column("median_income")
  rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
  # Divide households into 7 buckets.
  bucketized_households = tf.feature_column.bucketized_column(
    households, boundaries=get_quantile_based_boundaries(
      training_examples["households"], 7))

  # Divide longitude into 10 buckets.
  bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=get_quantile_based_boundaries(
      training_examples["longitude"], 10))
  
  # Divide latitude into 10 buckets.
  bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=get_quantile_based_boundaries(
      training_examples["latitude"], 10))

  # Divide housing_median_age into 7 buckets.
  bucketized_housing_median_age = tf.feature_column.bucketized_column(
    housing_median_age, boundaries=get_quantile_based_boundaries(
      training_examples["housing_median_age"], 7))
  
  # Divide median_income into 7 buckets.
  bucketized_median_income = tf.feature_column.bucketized_column(
    median_income, boundaries=get_quantile_based_boundaries(
      training_examples["median_income"], 7))
  
  # Divide rooms_per_person into 7 buckets.
  bucketized_rooms_per_person = tf.feature_column.bucketized_column(
    rooms_per_person, boundaries=get_quantile_based_boundaries(
      training_examples["rooms_per_person"], 7))
  
  feature_columns = set([
    bucketized_longitude,
    bucketized_latitude,
    bucketized_housing_median_age,
    bucketized_households,
    bucketized_median_income,
    bucketized_rooms_per_person])
  
  return feature_columns
train_model(
    learning_rate=1.0,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
'''
#特征组合
'''特征列 API 仅支持组合离散特征。要组合两个连续的值（比如 latitude 或 longitude），我们可以对其进行分桶'''
'''
def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """ 
  households = tf.feature_column.numeric_column("households")
  longitude = tf.feature_column.numeric_column("longitude")
  latitude = tf.feature_column.numeric_column("latitude")
  housing_median_age = tf.feature_column.numeric_column("housing_median_age")
  median_income = tf.feature_column.numeric_column("median_income")
  rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
  # Divide households into 7 buckets.
  bucketized_households = tf.feature_column.bucketized_column(
    households, boundaries=get_quantile_based_boundaries(
      training_examples["households"], 7))

  # Divide longitude into 10 buckets.
  bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=get_quantile_based_boundaries(
      training_examples["longitude"], 10))
  
  # Divide latitude into 10 buckets.
  bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=get_quantile_based_boundaries(
      training_examples["latitude"], 10))

  # Divide housing_median_age into 7 buckets.
  bucketized_housing_median_age = tf.feature_column.bucketized_column(
    housing_median_age, boundaries=get_quantile_based_boundaries(
      training_examples["housing_median_age"], 7))
  
  # Divide median_income into 7 buckets.
  bucketized_median_income = tf.feature_column.bucketized_column(
    median_income, boundaries=get_quantile_based_boundaries(
      training_examples["median_income"], 7))
  
  # Divide rooms_per_person into 7 buckets.
  bucketized_rooms_per_person = tf.feature_column.bucketized_column(
    rooms_per_person, boundaries=get_quantile_based_boundaries(
      training_examples["rooms_per_person"], 7))
  
  # YOUR CODE HERE: Make a feature column for the long_x_lat feature cross
  long_x_lat = tf.feature_column.crossed_column(
  set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000) 
  
  feature_columns = set([
    bucketized_longitude,
    bucketized_latitude,
    bucketized_housing_median_age,
    bucketized_households,
    bucketized_median_income,
    bucketized_rooms_per_person,
    long_x_lat])
  
  return feature_columns
'''


