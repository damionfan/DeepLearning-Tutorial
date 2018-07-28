import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''----------------Series 列表--------------------'''
s=pd.Series([1,3,5,np.nan,6,8])
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64
'''-------------DataFrame 时间索引 带有标记列 的numpy数组----------------'''
dates=pd.date_range('20180101',periods=6)
# DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
#                '2018-01-05', '2018-01-06'],
#               dtype='datetime64[ns]', freq='D')
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
#                    A         B         C         D
# 2018-01-01 -1.986327 -1.297630  0.036183 -0.898657
# 2018-01-02  0.964481 -0.021326  0.110256  0.466615
# 2018-01-03 -0.244967 -1.858919 -2.202228  0.663343
# 2018-01-04  0.268088  0.405183  0.519485 -0.734043
# 2018-01-05  0.907235  1.069966 -0.191850 -1.250235
# 2018-01-06  0.340554  0.341000 -0.437750  0.307312
'''DataFrame也可以用类似的dict来创建'''
df2=pd.DataFrame({'A':1.,
                'B':pd.Timestamp('20180101'),
                'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                'D':pd.Categorical(['长度必须','和上面的','range(4)','一样长'])})
#    A          B    C         D
# 0  1.0 2018-01-01  1.0      长度必须
# 1  1.0 2018-01-01  1.0      和上面的
# 2  1.0 2018-01-01  1.0  range(4)
# 3  1.0 2018-01-01  1.0       一样长

