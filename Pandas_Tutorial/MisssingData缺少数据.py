import numpy as np
import pandas as pd
from Selection选择 import df

dates=pd.date_range('20180101',periods=6)
'''使用np.nan表示缺省，不参与计算'''

df1=df.reindex(index=dates[0:4],columns=list(df.columns)+['E'])
df1.loc[dates[0]:dates[1],'E']=1
print(df1)
#                    A         B         C  D    F    E
# 2018-01-01  0.000000  0.000000 -2.256709  5  NaN  1.0
# 2018-01-02  1.779415 -1.614963 -0.151882  5  1.0  1.0
# 2018-01-03 -1.580324  2.736013  1.313186  5  2.0  NaN
# 2018-01-04  1.130595 -0.826144  0.070675  5  3.0  NaN
'''删除nan的行'''
dele=df1.dropna(how='any')#没有改变df1
print(dele)
#                    A         B         C  D    F    E
# 2018-01-02  0.306111 -0.493546 -0.246071  5  1.0  1.0
'''填充缺失数据'''
fill_=df1.fillna(value=5)#df1没变
print(fill_)
#                    A         B         C  D    F    E
# 2018-01-01  0.000000  0.000000 -0.554117  5  5.0  1.0
# 2018-01-02  0.390360  0.206235 -0.622870  5  1.0  1.0
# 2018-01-03  0.240348 -0.042710  1.110336  5  2.0  5.0
# 2018-01-04  1.101651  0.277547 -1.250565  5  3.0  5.0
'''获取布尔掩码'''
print(pd.isnull(df1))
#                 A      B      C      D      F      E
# 2018-01-01  False  False  False  False   True  False
# 2018-01-02  False  False  False  False  False  False
# 2018-01-03  False  False  False  False  False   True
# 2018-01-04  False  False  False  False  False   True