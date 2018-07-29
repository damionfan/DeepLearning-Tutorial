import numpy as np
import pandas as pd

'''--------------------------csv---------------------------'''
'写入csv'
df=pd.DataFrame(np.random.randn(4,4),index=pd.date_range('20180101',periods=4),columns=list('abcd'))

df.to_csv('data/foo.csv')
'''从csv中读入'''
read=pd.read_csv('data/foo.csv')
'''--------------------------HDF5-------------------------'''
# df.to_hdf('data/foo.h5','df')
# df.read_hdf('data/foo.h5','df')

'''-------------------------excel-----------------------'''
df.to_excel('data/foo.xlsx',sheet_name='Sheet1')
pd.read_excel('data/foo.xlsx','sheet1',index_col=None,na_values=['NA'])

