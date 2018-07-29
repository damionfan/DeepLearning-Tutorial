import numpy as np
import pandas as pd

np.random.seed(0)
'''--------------------根据要求把数据拆分-------------------
   --------------------把功能独立应用于组-------------------
   --------------------把结果组合到数据结构中--------------- '''
df=pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                'C' : np.random.randn(8),
                'D' : np.random.randn(8)})
print(df)
#      A      B         C         D
# 0  foo    one  1.764052 -0.103219
# 1  bar    one  0.400157  0.410599
# 2  foo    two  0.978738  0.144044
# 3  bar  three  2.240893  1.454274
# 4  foo    two  1.867558  0.761038
# 5  bar    two -0.977278  0.121675
# 6  foo    one  0.950088  0.443863
# 7  foo  three -0.151357  0.333674
'''根据A分组(groupby),应用sum'''
print(df.groupby('A').sum())
#             C         D
# A
# bar  1.663773  1.986547
# foo  5.409080  1.579400
'''多列分组，分层索引'''
print(df.groupby(['A','B']).sum())
#                   C         D
# A   B
# bar one    0.400157  0.410599
#     three  2.240893  1.454274
#     two   -0.977278  0.121675
# foo one    2.714141  0.340644
#     three -0.151357  0.333674
#     two    2.846296  0.905081


