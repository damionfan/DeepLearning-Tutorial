import pandas as pd
import numpy as np

# df=pd.read_csv('data/foo.csv')
# print(df)
# df2=df#视图 之后的pop也会改变df2，你懂的吧
# df3=df.pop('a')
# print(df2)
# print(df3)

df=pd.read_csv('data/foo.csv',names=[1,2,3,4],header=0)#这样可以改变原来的列名
print(df)
'''names=[0,1,2,3,4]的话就可以把索引变成一列数据'''