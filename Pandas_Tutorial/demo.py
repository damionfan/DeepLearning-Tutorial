import pandas as pd
import numpy as np

# df=pd.read_csv('data/foo.csv')
# print(df)
# df2=df#视图 之后的pop也会改变df2，你懂的吧
# df3=df.pop('a')
# print(df2)
# print(df3)

# df=pd.read_csv('data/foo.csv',names=[1,2,3,4],header=0)#这样可以改变原来的列名
# print(df)
# #                    1         2         3         4
# # 2018-01-01  0.426003 -0.483873  0.009144  1.231409
# # 2018-01-02 -0.396477  0.299704  0.856330  0.172590
# # 2018-01-03  0.077818 -1.632361 -0.997272  1.939307
# # 2018-01-04 -1.511099  1.043059  1.282157  0.698524
# '''names=[0,1,2,3,4]的话就可以把索引变成一列数据'''

pf=pd.read_csv('data/foo.csv',header=0)
pf=pf.as_matrix()
print(pf)