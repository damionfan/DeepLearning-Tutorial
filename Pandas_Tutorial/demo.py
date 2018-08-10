import pandas as pd
import numpy as np

df=pd.read_csv('data/foo.csv')
print(df)
df2=df#视图 之后的pop也会改变df2，你懂的吧
df3=df.pop('a')
print(df2)
print(df3)