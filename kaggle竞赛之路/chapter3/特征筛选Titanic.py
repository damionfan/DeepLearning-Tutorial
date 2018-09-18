# coding=utf-8
import pandas as pd
import numpy as np

titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
y=titanic['survived']
x=titanic.drop(['row.names','name','survived'],axis=1)
x['age'].fillna(x['age'].mean(),inplace=True)
x.fillna('unknown',inplace=True)

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=66)

#类别型特征向量化
from sklearn.feature_extraction import DictVectorizer

vec=DictVectorizer()
x_train=vec.fit_transform(x_train.to_dict(orient='record'))
x_test=vec.transform(x_test.to_dict(orient='record'))

#输出处理后的特征的维度
print(len(vec.feature_names_))
#决策树
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
dt.score(x_test,y_test)


#倒入特征提取筛选器
from sklearn import feature_selection

#筛选前20%的特征，使用决策树进行预测，并且评估性能
fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)
x_train_fs=fs.fit_transform(x_train,y_train)
x_test_fs=fs.transform(x_test)
dt.fit(x_train_fs,y_train)
dt.score(x_test_fs,y_test)

#交叉验证 按固定百分比筛选特征，并且plot
from sklearn.cross_validation import cross_val_score

percentiles=range(1,100,2)

results=[]
for i in percentiles:
    fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    x_train_fs=fs.fit_transform(x_train,y_train)
    scores=cross_val_score(dt,x_train_fs,y_train,cv=5)#交叉验证 几折 默认3
    results=np.append(results,scores.mean())

print(results)

#找到最好的
opt=np.where(results==results.max())[0]
print('max opt:%d'%percentiles[int(opt)])

import matplotlib.pyplot as plt

plt.plot(percentiles,results)
plt.xlabel('percentiles of features')
plt.ylabel('acc')
plt.show()

#找到最好的之后在test上性能评估
from sklearn import feature_selection
fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=7)
x_train_fs=fs.fit_transform(x_train,y_train)
dt.fit(x_train_fs,y_train)
x_test_fs=fs.transform(x_test)
dt.score(x_test_fs,y_test)
