import xgboost as xgb
import numpy as np
'''-------数据接口---------
libsvm 格式
numpy 2-D !的数组
xbgoost二进制缓冲文件
存在DMatrix的对象中
'''
#libsvm
dtrain=xgb.Dmatrix('trian.svm.txt')
#XGBoost 二进制文件
dtest=xgb.DMatrix('test.svm.buffer')
#numpy 2-D
data=np.random.rand(5,10)#5 entities each contains 10 features
label=np.random.randint(2,size=5)
dtrain=xgb.DMatrix(data=data,label=label)
