import xgboost as xgb
#计算分类正确率
from sklearn.metrics import accuracy_score
'''
二、数据读取
XGBoost可以加载libsvm格式的文本数据，libsvm的文件格式（稀疏特征）如下：
1  101:1.2 102:0.03
0  1:2.1 10001:300 10002:400
...
每一行表示一个样本，第一行的开头的“1”是样本的标签。“101”和“102”为特征索引，'1.2'和'0.03' 为特征的值。
在两类分类中，用“1”表示正样本，用“0” 表示负样本。也支持[0,1]表示概率用来做标签，表示为正样本的概率。

下面的示例数据需要我们通过一些蘑菇的若干属性判断这个品种是否有毒。
UCI数据描述：http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/ ，
每个样本描述了蘑菇的22个属性，比如形状、气味等等（将22维原始特征用加工后变成了126维特征，
并存为libsvm格式)，然后给出了这个蘑菇是否可食用。其中6513个样本做训练，1611个样本做测试。

注：libsvm格式文件说明如下 https://www.cnblogs.com/codingmengmeng/p/6254325.html

XGBoost加载的数据存储在对象DMatrix中
XGBoost自定义了一个数据矩阵类DMatrix，优化了存储和运算速度
DMatrix文档：http://xgboost.readthedocs.io/en/latest/python/python_api.html

'''
dir='mushroom_data/'
dtrain=xgb.DMatrix(dir+'agaricus.txt.train')
dtest=xgb.DMatrix(dir+'agaricus.txt.test')

'''
describe for date :
dtrain.num_col()
dtrain.num_row()
...
训练参数：
max_depth:树的最大深度，缺省值为6 取值范围[1,infinite]
eta:防止过拟合，更新过程中用到的收缩步长，每次提升计算之后，算法会直接获得新特征的权重
    eta 通过缩减特征的权重使提升过程更加保守，缺省0.3， 取值范围[0,1]
silent: 取值为0时 表示打印出运行的信息，取值为1：表示以缄默的的方式运行，不打印运行信息，默认0
objective： 定义学习的任务和相应的学习目标，‘binary:logistic"表示二分类的逻辑回归为题，输出为概率

'''
#param  map
param={
    'max_depth':2,
    'eta':1,
    'silent':0,
    'objective':'binary:logistic',

}

#设置boosting 迭代次数
num_round=2;
import time
start_time=time.time()
#train
bst=xgb.train(param,dtrain,num_round)
#output:probability
#probability -> 0/1
train_preds=bst.predict(dtrain)
train_predictions=[round(value) for value in train_preds]# ceil 向上取整 round:四舍五入

y_train=dtrain.get_label()#输入数据的第一行
train_accuracy=accuracy_score(y_train,train_predictions)
print('train acc : %.2f %%'%(train_accuracy*100))
#test
preds=bst.predict(dtest)
predicts=[round(value) for value in preds]
y_test=dtest.get_label()
test_accuracy=accuracy_score(y_test,predicts)
print('test acc :%.2f %%'%(test_accuracy*100))

#可视化
'''
 使用XGBoost工具包里面的plot_tree （需要graphviz软件包)
 plot_tree() params:
 1 :模型
 2 : 树的索引，从0开始
 3 : 显示反向，默认竖直 "LR"是水平方向
'''
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import matplotlib.pyplot as plt
# import graphviz #这个需要先安装一个msi在windows上 其他的暂时没用到
# xgb.plot_tree(bst,num_trees=0,rankdir='LR')
# plt.show()
