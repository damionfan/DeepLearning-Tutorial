import keras as ks
from keras.models import Sequential
from keras.layers import Dense,Activation
'''-----------------------快速开始的Sequential模型----------------------'''
model=Sequential([
    Dense(units=32,input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
#alternative
model1=Sequential()
model1.add(Dense(32,input_dim=784))
model1.add(Activation('relu'))
model1.add(Dense(10))
model1.add(Activation('softmax'))

'''----------------------------------指定输入数据的shape-----------------------


模型需要知道输入数据的shape,因此Sequential的第一层接收一个关于关于输入数据的shape的参数，
后面的层 可以自己推倒,
几种方法为第一层指定输入数据的shape:
1.传递一个input_shape 的关键词参数给第一层，input_shape 是tuple ,也可是None,None:可能是任何正整数,d
但是！数据的batch_size 不应该包含在input_shape其中
2. batch_input_shape 关键字参数，包含batch，这个在指定固定batch时比较有用。
    实际上：keras在内部会添加一个None 在input_shape 来转化为batch_input_shape
3. 2D层！eg:Dense,可以通过输入维度input_dim 来隐含指定输入的shape,
3. 3D ！ 支持通过参数input_dim 和 input_length 来指定输入的shape
'''
# 下面3个是等价的
#1
model1.add(Dense(32,input_shape=(784,)))#tuple
#2 have batch!
model1.add(Dense(32,batch_input_shape(None,784)))
#3
model1.add(Dense(32,input_dim=784))
# 下面也是
#1
model1.add(LSTM(32,input_dim=(10,64)))
#2
model1.add(LSTM(32,batch_input_dim=(None,10,64)))
#3
model1.add(LSTM(32,input_length=10,input_dim=64))

'''------------------------------Merge层
多个Sequential 可以经由一个Merge层合并到一个输出，Merge层的output是一个可以被添加到Sequential的层对象
'''
#把两个Sequential合并到一起
from keras.layers import Merge

left_branch=Sequential()
left_branch.add(Dense(32,input_dim=784))

right_brance=Sequential()
right_brance.add(Dense(32,batch_input_shape=(None,784)))

merged=Merge([left_branch,right_brance],mode='concat')
'''
mode:
sum(default):逐元素相加
concat:张量串联，可以通过cancat_axis的关键词参数指定按照哪个轴进行串联 可以理解沿着concat_axis这个轴的变长
mul:元素相乘
ave:张量平均
dot:张量相乘，可以通过dot_axis 关键词参数指定要消去的轴
cos:计算2D张量（矩阵）中各向量的余弦距离
alternative:
可以提供关键字参数，以实现任意的变换 ，？？？？？？？？？？？？？？？？？？
merged=Merge([left,right],mode=lambda x:x[1]-x[0])
'''
final_model=Sequential()
final_model.ad(merged)#!!!!!!!!!!!!!!
model.add(Dense(10,activation='softmax'))

#分支模型可以通过下面的代码训练
final_model.compile(optimizer='rmsprop',loss='categorial_crossentropy')
final_model.fit([input_data_1,input_data_2],targets)#pass one data array per model input

'''---------------------------编译----------------------------------
接收3个参数：
optimizer:rmsprop,adagrad,或者是一个OPtimizer类的对象
loss函数:categorial_crossentropy,mse/也可是一个损失函数
指标列表metrics:对于分类问题，我们把列表设置为metrics=['accuracy'],指标可以是一个预定义的名字，也可是一个用户定制的函数
指标函数应该返回单个张量或者是完成metrc_name_>metirc_value的映射字典，参考性能评估
'''
#for a multi-class classification problem
model.compile(optimizer='rmsprop',loss='categorial_crossentropy',metrics=['accurcay'])
#for a binary classification problem
model.comile(optimizer='rmsprop',loss='binary_crosentropy',metrics=['accuracy'])
#for a mean squared error regression problem
model.compile(optimizer='rmsprop',loss='mse')
#for custon metrics
import keras.backend as K
def mean_pred(y_true,y_pred):
    return K.mean(y_pred)
def false_rate(y_ture,y_pre):
    false_neg=...#negative
    false_pos=...#positive
    return {
        'false_neg':false_neg,
        'false_pos':false_pos,
    }
#把一个函数传递进去
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy',mean_pred,false_rate])
'''----------------------------训练-----------------------------'''
#for a single-input model with 2 classes (binary)
model=Sequential()
model.add(Dense(1,input_dim=784,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#generate dummy data dummy=fake
import numpy as np
data=np.random.ranom((1000,784))
labels=np.random.randint(2,size=(1000,1))#Return random integers from `low` (inclusive) to `high` (exclusive).
model.fit(data,labels,batch_size=10,nb_epoch=10)

#for a multi-input model with 10 classes
left=Sequential(Dense(32,input_dim=784))
right=Sequential(Dense(32,input_dim=784))

merged=Merge([left,right],mode='concat')
model=Sequential(merged)
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorial_crossentropy',metrics=['accuracy'])

#generate dummy data
data1=np.random.random((1000,784))
data2=np.random.random((1000,784))
labels=np.random.randint(10,size=(1000,1))
#convert labels ->binary matrix of size (1000,10)
from keras.utils.np_utils import to_categorical #just lile pd.get_dummies()
labels=to_categorical(y=labels,num_classes=10)
'''y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.'''
#train
model.fit([data_1,data_2],labels,np_epoch=10,batch_size=32)




