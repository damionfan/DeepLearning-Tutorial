# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#标准化标签 ->range()
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('mushrooms.csv')
# print('Dataset shape',df.shape)#(8214,23)
# print(df.dtypes)#object
le=LabelEncoder()
dataset=df.apply(le.fit_transform)#把'p'这些特征值变成数值，

targets=dataset['class']
columns=dataset.columns
data=pd.DataFrame(dataset['class'])
for i in columns:
    x=pd.get_dummies(dataset[i])#->one-hot
    data=data.join(x,lsuffix='_left',rsuffix='_right')
    #suffix :后缀
#num_example * 119 get_dummies 编码成one-hot

features=data.drop('class',axis=1)
# print(features.shape)#(8124,119)
features=features.values


'''
train_test_split(*$1,*$2)
$1:lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    :要有相同的length/shape[0]

'''
X_train,X_test,Y_train,Y_test=train_test_split(features,targets,
                                               test_size=0.2,random_state=0)
eval_set=[(X_test,Y_test)]

model=xgb.XGBClassifier()
'''

'''
#
model.fit(X_train,Y_train ,eval_metric='error',eval_set=eval_set,verbose=False)

pred=model.predict(features)
socre=accuracy_score(targets,pred)
print('acc score ',socre*100)