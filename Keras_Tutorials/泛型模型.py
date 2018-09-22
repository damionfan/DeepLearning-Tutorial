# coding=utf-8
import  numpy as np
import  pandas as pd
import keras
from keras.models import Model
from keras.layers import Input,Dense

a=Input(shape=(32,))
b=Dense(32)(a)
model=Model(input=a,output=b)
#也可以构造多个输入和多个输出的model
# model1=Modle(inputu=[a1,a2,a3],ouput=[b1,b2,b3])
'''
----------常用的Model的属性！：
model.layers
model.inputs
model.outputs

-----------------------------------Model的模型方法:compile---------------------
compile:
compile(self,optimizer,loss,metrics=[],loss_weights=None,sample_wieght_mode=None)
Args:
optimizer:optimizer
loss:object loss
metircs:列表，在train和test时的性能列表一般是metrics=['accuracy'];
    在多个输出模型中为不同的的输出指定不同的指标，可以使用字典metrics={'output_a':'accuracy'}
sample_weight_mode: 默认None
    当需要按时间步为样本赋值（2D权矩阵），则可以设为temporal（时间的）。
    None：按样本赋权（1D权）。
    多个模型输出可以使用字典
kwargs:使用tensorflow 的忽略

TIPS:
    如果你只是 载入 模型来实现predict，可以不用使用compile
    compile :完成loss函数和优化器的配置，for train.
    在predict 会在内部进行符号函数的编译工作（通过调用_make_predict_function生成函数）
    
    
-----------------------------------------fit------------------------------------
fit(self,x,y,batch_size=32,nb_epoch=10,verbose=1,callbacks=[],validation_spilt=0.0,
    validation_data=None,shuffle=True,class_weight=None,sample_weight=None
    
params:
x:input_data
    一个输入：dtype:numpy array
    多个输入：dtype:list,元素对应各个输入的numpy array
    模型输入有名字：那么就可以使用字典，把输入名字和输入数据对应
y:label ，numpy array ,
    多个输出：list of numpy array
    输出有名字：字典
batch_size:
    one batch one op
nb_epoch:
    int，训练轮数，data被遍历nb_epoch次,keras :nb :number of 
verbose:（啰嗦的），日志显示，
    0：不在标准输出流中输出日志信息，
    1：输入进度条记录
    2：为每个epoch输出一行记录
callbacks:list ,元素是keras.callbacks.Callback的对象。
    这个list的回调函数将在训练过程中适当时机被调用。
    回调函数：在训练的特定阶段被调用的函数集，可使用回调函数观察训练过程中内部的状态和统计信息。适用于fit（）
    （先稍微举几个例子有点概念：History,ModelCheckpoint,EearlyStopping,TensorBoard）
    tips:回调'函数'其实是一个类，只是习惯这么叫
validation_split: flaot :0-1 ,split train_data as validation_data ,在每个epoch后城市model的指标，如loss function，accuracy等
validation_data: (x,y)或者是(x,y,sample_weights)的tuple ，指定验证集，这个将覆盖validation_split !

shuffle:boolean
class_weight：dict 字典，将不同的类别映射到不同的权值，
        该参数用来在训练过程中调整loss_function（only in train op），
        在处理非平衡的训练数据（某些类的训练样本书很少），可以使得loss_function对样本不足的数据更加关注
sample_weight: 权值 numpy array，
        用于在train中调整loss_function（only for train op）
        传递一个1D的与样本等长的向量用于对样本进行1对1的加权，
        或者是在面对时序数据时传递一个（samples,sequence_length）的矩阵为每个时间步的样本赋不同的权值，这需要sample_weight='temporal'
    
            
    fit返回一个History的对象，其中History.history属性记录了loss_function和其他指标的数值随着epoch的变化情况，
                        如果有validation_data，也包括了validation的
    
'''
'''--------------------------evaluate----------------------
evaluate(self,x,y,batch_size=32,verbose=1,sample_weight=None)
Args:
    same with fit()
返回一个测试误差的标量值（如果模型没有其他的评价指标），或者是一个list（有其他的指标）。
model.metrics_names 给出list的各个值的含义
but: verbose :only 0/1 meaning same with fit()
'''
'''----------------------predict----------------------
predict(self,x,batch_size=32,verbose=0)
返回：预测是的numpy array
'''
'''-----------------------train/predict/test_on_batch-----------------
train_on_batch(self,x,y,class_weight=None,sample_weight=None)在一个 batch 的数据上进行 一次 参数更新
返回train_loss的list/scalar ,same with evaluate
test_on_batch(self,x,y,sample_weight=None) 在一个batch的样本上对model进行评估
返回与evaluate相同
predict_on_batch(self,x)在一个batch的样本上对模型进行测试
函数返回模型在一个Batch上的预测结果
'''
'''---------------------------fit_generator---------------------
fit_generator(self,generator,smaples_per_epoch,np_epoch,verbose=1,call_backs=[],validation_data=None,nb_val_samples=None,class_weight={},max_q_size=10)
利用python的生成器，逐个生成数据的batch并进行训练，生成器与模型将并行执行以提高效率
ex:函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型的训练
Args:
generator:生成器函数，生成器的输出应该为：
                                    一个形如（inputs,targets）的tuple
                                    一个形如（inputs,targets,sample_weights)的tuple.所有的返回值都应该是包含相同的数据的样本。
                                    生成器将在无限在数据集上循环。每个epoch以经过模型的样本数达到sample_per_epoch时，记一个epoch结束
sample_per_epoch:整数，等模型处理的样本打到此数目时计一个epoch结束，执行下一个epoch
verbose：0：不在标准输出流输出日志，1 输出进度条记录，2 每个epoch输出一条记录
validation_data :具有一下三个形式之一
    生成验证集的生成器
    一个形如（inputs,targets）的tuple
    一个形如（inputs,targets,sample_weights）的tuple
nb_val_samples:only valdiation_data是生成器时使用
                    用以限制在每个epoch结束时用来验证模型的验证集样本书，功能类似与sample_per_epoch
max_q_size:生成器队列的最大容量
返回一个History对象
examples:
def generate_arryas_from_file(path):
    while 1:
    f=open(path)
    for line in f:
        #create numpy arrays of input data
        #and labels,from each line in file
        x,y=process_line(line)
        yield(x,y)
    f.close()
model.fit_generator(generate_arrars_from_file('/my_file.txt'),
                    samples_per_epoch=10000,nb_epoch=10)
    
'''
'''--------------------evaluate_generator--------------------
evaluate_generator(self,gnenrator,val_samples,max_q_size=10)
使用一个生成器作为数据源，来评估模型，生成器应该返回与test_per_batch的输入数据相同类型的数据
Args:
    generator:生成输入batch数据的生成器
    val_samples:生成器应该返回的总的样本数
    max_q_size：生成器队列的最大容量
    nb_worker:使用基于进程的多线程处理时的进程数
    pickle_safe:True:使用基于进程的线程，注意因为他的实现依赖于多进程处理。不可传递不可pickle的参数到生成器汇总，因为他们不能轻易的传递到子进程中
    
'''
'''-----------------predict_generator----------------
predict_generator(self,generator,val_samples,max_q_size=10,nb_worker=1,pickle_safe=False)
从一个生成器上获取数据并进行预测，生成器应该返回与predict_on_batch输入类似的数据
Args:
    generator:生成输入Batch数据的生成器
    val_samples:生成器应该返回的总样本数
    max_q_size：生成器队列的最大容量
    
'''
'''-------------------get_layer-------------------
'get_layer(self,name=None,index=None)
本函数一句模型中层的下标或者是名字获得层对象，泛型模型的层的下标是从底向上，水平遍历的顺序
Args:
    name:name of layer
    index :index of layer
return:
    层对象
'''

