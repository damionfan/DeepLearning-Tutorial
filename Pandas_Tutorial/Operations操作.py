import numpy as np
import pandas as pd
from Selection选择 import df

dates=pd.date_range('20180101',periods=6)

'''统计'''
print(df.mean())
# A   -0.029139
# B   -0.241363
# C    0.582015
# D    5.000000
# F    3.000000
# dtype: float64
'''另一个轴'''
print(df.mean(1))
# 2018-01-01    1.296622
# 2018-01-02    0.850654
# 2018-01-03    1.551481
# 2018-01-04    2.043344
# 2018-01-05    1.632430
# 2018-01-06    2.236133
# Freq: D, dtype: float64
'''自动沿指定维度进行广播'''
