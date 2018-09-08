# coding=utf-8
import pandas as pd
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
    x=pd.get_dummies(dataset[i])
    data=data.join(x,lsuffix='_left',rsuffix='_right')
features=data.drop('class',axis=1)
# print(features.shape)#(8124,119)
features=features.values