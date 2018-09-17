# coding=utf-8
import  pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# print(titanic.head())
# NaN 缺失 数值型，类别 都不一样
# print(titanic.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 11 columns):
# row.names    1313 non-null int64
# pclass       1313 non-null object
# survived     1313 non-null int64
# name         1313 non-null object
# age          633 non-null float64
# embarked     821 non-null object
# home.dest    754 non-null object
# room         77 non-null object
# ticket       69 non-null object
# boat         347 non-null object
# sex          1313 non-null object
# dtypes: float64(1), int64(2), object(8)
# memory usage: 112.9+ KB

#预处理
x=titanic[['pclass','age','sex']]
y=titanic['survived']
# print(x.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 3 columns):
# pclass    1313 non-null object
# age       633 non-null float64 不够！！
# sex       1313 non-null object
# dtypes: float64(1), object(2)
# memory usage: 30.9+ KB

#age 不够长
x['age'].fillna(x['age'].mean(),inplace=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=66)

#特征提取
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)

x_trian=vec.fit_transform(x_train.to_dict(orient='record'))
print(vec.feature_names)
x_test=vec.transform(x_test.to_dict(orient='record'))


