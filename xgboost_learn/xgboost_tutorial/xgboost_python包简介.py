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
#scpiy .aparse 数组
csr=scipy.sparse.csr_matrix((dat,(row,col)))
dtrain=xgb.DMatrix(csr)
#保存DMatrix到XGBoost的二进制文件 下次加载更快
dtrain=xgb.DMatrix('trian.svm.txt')
dtrain.save_binary('train.buffer')

#处理DMatrix 里的缺失值 可以指定缺失值的参数来初始化DMatrix
dtrain=xgb.DMatrix(data,label=label,missing=-999.0)

#可是设置权重
w=np.random.rand(5,1)
dtrian=xgb.DMatrix(data,label=label,missing=-1,weight=w)#???

#设置参数
#使用pair格式
'''
训练参数：
max_depth:树的最大深度，缺省值为6 取值范围[1,infinite]
eta:防止过拟合，更新过程中用到的收缩步长，每次提升计算之后，算法会直接获得新特征的权重
    eta 通过缩减特征的权重使提升过程更加保守，缺省0.3， 取值范围[0,1]
silent: 取值为0时 表示打印出运行的信息，取值为1：表示以缄默的的方式运行，不打印运行信息，默认0
objective： 定义学习的任务和相应的学习目标，‘binary:logistic"表示二分类的逻辑回归为题，输出为概率

'''
param={
    'bst_max_depth':2,
    'bst:eta':1,
    'silent':1,
    'objective':'binary:logistic'
}
param['nthread']=4
param['eval_metric']='auc'#area...
#
