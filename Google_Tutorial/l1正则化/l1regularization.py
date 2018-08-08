
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

'''计算模型大小'''
# ，只需计算非零参数的数量即可
def model_size(estimator):
    variables=estimator.get_variable_names()
    size=0
    for variable in variables:
        if not any(x in variable
                   for x in ['global_step','centered_bias_weight','bias_weight','Ftrl']):
            size+=np.count_nonzeor(estimator.get_variable_value(variable))

    return size

model_size(linear_classifier)

'''l1'''
my_optimizer=tf.train.FtrlOptimizer(learning_rate,l1_regularization_strength=regularization_strength)