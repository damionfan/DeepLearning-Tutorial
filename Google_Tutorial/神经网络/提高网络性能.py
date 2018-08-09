import math
from matplotlib import cm,gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    processed_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] /
        california_housing_dataframe["population"])
    return processed_features
def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = (
        california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets

training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

'''DNNRegressor,默认relu,fc'''
def contruct_feature_columns(input_feature):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_feature])

def my_input_fn(features,targets,batsh_size=1,shuffle=True,num_epoch=None):
    features={key:np.array(value)
              for key,value in dict(features).items()}
    ds=Dataset.from_tensor_slices((features,targets))
    ds=ds.batch(batch_size).repeat(num_epoch)
    if shuffle:
        ds=ds.shuffle(10000)
    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels
def train_nn_model(my_optimizer,steps,batsh_size,hidden_units,
                   training_examples,training_targets,validation_examples,validation_targets):
    periods=10
    steps_per_period=steps/periods
    my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
    dnn=tf.estimator.DNNClassifier(hidden_units,
                                   feature_columns=contruct_feature_columns(training_examples),
                                   optimizer=my_optimizer)
    #input fn
    train_input_fn=lambda:my_input_fn(training_examples,training_targets['median_house_value'],batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value"],
                                                      num_epochs=1,
                                                      shuffle=False)
    #train
    print('train')
    train_rmse=[]
    validation_rmse=[]
    for period in range(0,periods):
        dnn.train(input_fn=train_input_fn,steps=steps_per_period)
        #predict
        t_predictions=dnn.predict(predict_training_input_fn)
        t_predictions=np.array([item['predictions'][0]
                                for item in t_predictions])
        v_predictions=dnn.predict(predict_validation_input_fn)
        v_predictions=np.array([item['predictions'][0]
                                for item in v_predictions])
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        print("  period:train_RMSE %02d : %0.2f" % (period, training_root_mean_squared_error))
        train_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(train_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn, train_rmse, validation_rmse

train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

'''输入标准化以使其位于 (-1, 1) 范围内可能是一种良好的标准做法 归一化'''
def linear_scale(s):
    min_v=s.min()
    max_v=s.max()
    scale=(max_v-min_v)/2.0
    return s.apply(lambda x:(x-min_v)/scale -1.0)

def normalize_linear_scale(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
  processed_features = pd.DataFrame()
  processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
  processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
  processed_features["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
  processed_features["total_bedrooms"] = linear_scale(examples_dataframe["total_bedrooms"])
  processed_features["population"] = linear_scale(examples_dataframe["population"])
  processed_features["households"] = linear_scale(examples_dataframe["households"])
  processed_features["median_income"] = linear_scale(examples_dataframe["median_income"])
  processed_features["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])
  return processed_features
