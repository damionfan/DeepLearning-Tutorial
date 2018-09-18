# coding=utf-8

import pandas as pd
import numpy as np

column_names=[
    'Sample code number','Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses','Class'
]
data=pd.read_csv('../breast-cancer-wisconsin.data.txt',names=column_names)
# ？-> np.nan
data=data.replace(to_replace='?',value=np.nan)
#drop np.nan
data.dropna(how='any',inplace=True)
#inplace 就地进行operation

# print(data.shape)#(699,11) class 2/4

#25% for validation 75% for train
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

# print(y_train.value_counts(),y_test.value_counts())

# 2    341
# 4    183
# Name: Class, dtype: int64
# 2    117
# 4     58
# Name: Class, dtype: int64

#使用sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

#标准化数据：保证每个维度方差为1，mean=0,
ss=StandardScaler()
x_train=ss.fit_transform(x_train)#== fit() transform()
x_test=ss.transform(x_test)

#init logisticRegression and SGDClassifier
lr=LogisticRegression()
sgdc=SGDClassifier()
#use fit() of L...
lr.fit(x_train,y_train)
#predict()
lr_y_pred=lr.predict(x_test)


#use fit() of S..
sgdc.fit(x_train,y_train)
sgdc_y_pred=sgdc.predict(x_test)

#性能分析
from sklearn.metrics import classification_report
# acc of LR C...
# print('acc of LR Classifier :',lr.score(x_test,y_test))
# print(classification_report(y_test,lr_y_pred,target_names=['良','恶']))
# acc of LR Classifier : 0.9883040935672515
#              precision    recall  f1-score   support
#
#           良       0.99      0.99      0.99       100
#           恶       0.99      0.99      0.99        71
#
# avg / total       0.99      0.99      0.99       171
# acc of SGD C...
# print('acc of SGD Classifier :',sgdc.score(x_test,y_test))
# print(classification_report(y_test,sgdc_y_pred,target_names=['良','恶']))
# acc of SGD Classifier : 0.9473684210526315
#              precision    recall  f1-score   support
#
#           良       0.92      1.00      0.96       100
#           恶       1.00      0.87      0.93        71
#
# avg / total       0.95      0.95      0.95       171


