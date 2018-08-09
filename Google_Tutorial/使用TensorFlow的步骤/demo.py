import pandas as pd

df=pd.read_csv('california_housing_train.csv',sep=',')
print(df.head(5))

df1=pd.read_csv('california_housing_train.csv',sep=',',header=None)
print(df1.head(5))
df2=pd.read_csv('california_housing_train.csv',sep=',',header=1)
print(df2.head(5))
'''header=None 没有列名，把所有的都当作数据，即使有列名，也把原来的列名作为数据，然后index=0
 header=0 表明数据开始的行数！，这里行数是从0开始，而且是DataFrame原来的index
 eg:有列名：header=0，数据从index=0开始，和默认的一样，header=1,数据从index=1开始，那么index=0的数据变为列名'''