import lightgbm as lgb
import numpy as np
import scipy


'''-------------数据接口------------
libsvm/tsv/csv txt format file（libsvm/tsv/csv 文本文件格式）
Numpy 2D array, pandas object（Numpy 2维数组, pandas 对象）
LightGBM binary file（LightGBM 二进制文件）
加载后的数据存在 Dataset 对象中.
'''
#ligsvm/lightGBM 二进制
trian_data=lgb.Dataset('train.svm.bin')
#numpy
data=np.random.rand(500,10)
label=np.random.randint(2,size=500)
trian_data=lgb.Dataset(data,label=label)
#scpiy.sparse_csr_matrix

csr=scipy.sparse.csr_matrix((dat,(row,col)))
train_data=lgb.Dataset(csr)
#保存Dataset -> lightGBM bin 加载更快
train_data=lgb.Dataset('train.svm.txt')
train_data.save_binary('train.bin')

#创建验证数据
test_data=train_data.create_valid('test.svm')
#or
test_data=lgb.Dataset('test.svm',reference=train_data)

#指定feature names  和 categorical features(分类特征/离散特征 dtype=object)
train_data=lgb.Dataset(data,label=label,feature_name=['c1','c2','c3'],categorical_feature=['c3'])
#lightGBM 可以直接使用categorical_feature 作为input,不需要转换成one_hot coding ，而且速度更快
#tips: 在之前要把categorical_feature 转换为int类型

#设置权重
w=np.random.rand(500,)
train_data=lgb.Dataset(data,label=label,weight=w)
#or
train_data.set_weight(w)

#Dataset_set_init_score() 可是初始化分数，以及可以使用Dataset.set_group()来设置group/query 数据以用于ranking(排序)任务

#内存的高使用 ....

'''-----------------------设置参数----------------------------'''
#使用pair的list或者是一个字典来设置参数
#Booster（提升器）参数
param={
    'num_leaves':31,
    'num_trees':100,
    'objective':'binary'}
param['metric']='auc'

#指定多个eval的指标
param['metric']=['auc','binary_logloss']

'''---------------------------训练-------------------------_'''
#训练一个模型需要，一个parameter list 和dataset
num_round=10
bst=lgb.train(param,train_data,num_round,valid_set=[test_data])

#保存模型
bst.save_model('model.txt')
#convert to >JSON
json_model=bst.dump_model()

#load the save model
bst=lgb.Booster(model_file='model.txt')

'''---------------------------交叉验证------------------------'''
#5折
num_round=10
lgb.cv(params=param,train_set=train_data,num_round=num_round,nfold=5)

'''-----------------------提前停止-------------------------------------'''
valid_sets=lgb.Dataset()
#如果有验证集，可以使用提前停止找到最佳的boosting rounds（提升次数）。提前停止需要在valid-sets中至少有一个集合，如果有多个他们都会使用
bst=lgb.train(param,train_data,num_round,valid_sets=valid_sets,early_stopping_rounds=10)
bst.save_model('model.txt',num_iteration=bst.best_iteration)
#模型将开始训练，直到验证得分停止提高为止，验证错误需要至少每个early_stopping_rounds减少 以继续训练
#如果提前停止，模型将有一个额外的字段bst.best_iteration.
#train()将返回最后一次接待的模型，不是最好的一个
#如果指定了多个metrics他们都被用于提前停止，metrics minimize(l2,log loss,etc) and to maximize(AUC,NDCG)
'''------------------------预测----------------------------------------------'''
ypred=bst.predict(data)
# 如果使用了提前停止，可以使用best.iteration 获得最佳的结果
ypred=bst.predict(data,num_iteration=bst.best_iteration)










