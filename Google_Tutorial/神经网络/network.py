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

def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
    features={key:np.array(value)
              for key,value in dict(features).items()}
    ds=Dataset.from_tensor_slices((features,targets))
    ds=ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds=ds.shuffle(10000)
    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels

def train_nn_regression_model(learning_rate,steps,batch_size,hidden_units,
                              training_examples,trainging_targets,
                              validation_examples,validation_targets):
    periods=10 #要记录多少次
    steps_per_period=steps/periods

    #regressor
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
    dnn=tf.estimator.DNNRegressor(hidden_units=hidden_units,
                                  feature_columns=contruct_feature_columns(training_examples))

    #input fn
    training_input_fn=lambda:my_input_fn(training_examples,training_targets['median_house_value'],batch_size=batch_size)
    predict_train_fn=lambda:my_input_fn(training_examples,trainging_targets['median_house_value'],shuffle=False,num_epochs=1)
    predict_validation_fn=lambda:my_input_fn(validation_examples,validation_targets['median_house_value'],shuffle=False,num_epochs=1)

    #train
    train_rmse=[]
    validation_rmse=[]
    for periods in range(0,periods):
        dnn.train(input_fn=training_input_fn,steps=steps_per_period)

        #validation by predicting
        train_predictions=dnn.predict(input_fn=predict_train_fn)
        train_predictions=np.array([item['predictions'][0]
                                    for item in train_predictions])
        validation_predictions=dnn.predict(input_fn=predict_validation_fn)
        validation_predictions=np.array([item['predictions'][0]
                                        for item in validation_predictions])
        #loss
        tr_=math.sqrt(metrics.mean_squared_error(train_predictions,trainging_targets))
        va_=math.sqrt(metrics.mean_squared_error(validation_predictions,validation_targets))

        train_rmse.append(tr_)
        validation_rmse.append(va_)


    print('model training finished')

    #plot
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(train_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn

dnn=train_nn_regression_model(0.01,500,10,[10,2],
                              training_examples,training_targets,
                              validation_examples,validation_targets)
plt.show()