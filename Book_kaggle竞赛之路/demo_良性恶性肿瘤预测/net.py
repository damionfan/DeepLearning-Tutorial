# coding=utf-8
#因为没有dataset所没法跑
import pandas as pd
import matplotlib.pyplot as plt

df_train=pd.train_csv('../Dataset/Breast-Cancer-trina.csv')
df_test=pd.read_csv('test.csv')
#选取clump thickness 'cell size' 作为特征
df_test_negative=df_test.loc[df_test['type']==0][['clump thickness','cell size']]
df_test_postive=df_test.loc[df_test['type']==1][['clump thickness','cell size']]

plt.scatter(df_test_negative['clump thickness'],df_test_negative['cell size'],marker='0',s=200,c='red')
#marker 是散点的形状
'''
nothing to learn so quit()
'''