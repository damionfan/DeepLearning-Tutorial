# coding=utf-8
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import numpy as np

boston=load_boston()
x=boston.x
y=boston.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=33,test_size=0.25)

#np.max np.min np.mean

