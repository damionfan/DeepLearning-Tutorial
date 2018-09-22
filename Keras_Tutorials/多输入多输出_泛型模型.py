# coding=utf-8
from keras.layers import Input,Embedding,LSTM,Dense,merge
from keras.models import Model
#headline input :meant to receive sequences of 100 integers,between 1 and 100000.
#note that we can name any layer by passing it a 'name 'argument
main_input=Input(shape=(100,),dtype='int32',name='main_input')

#thisi embedding layer will encode the input sequence
#into a sequence of dense 512-dimensional vectors
x=Embedding(output_dim=512,input_dim=10000,input_length=100)

#a LSTm will transform the vector sequence into a single vector
#containng information about the entire sequence
lstm_out=LSTM(32)(x)

#之后我们插入一个额外的损失，即使在🐷loss很高的情况下，LSTM和Embedding也是可以很好的训练
auxiliary_out_put=Dense(1,activation='sigmoid',name='aux_ouput')(lstm_out)

#然后我们吧lstm与额外的输入数据串联起来组成输入，送到模型中
auxiliary_input=Input(shape=(5,),name='aux_iput')
x=merge([lstm_out,auxiliary_input],mode='concat')

#we stack a deep fc network on top
x=Dense(64,activation='relu')(x)
x=Dense(64,activation='relu')(x)
x=Dense(64,activation='relu')(x)

#and finally we add the main logistic regression layer
main_output=Dense(1,activation='sigmoid',name='main_output')(x)

#最后定义整个2输入，2输出 的模型
model=Model(inputs=[main_input,auxiliary_input],output=[main_output,auxiliary_out_put])

#下一步编译，给 额外的损失 赋0.2的权重，我们可以通过关键词loss_weight/loss 为不同的输出设置不同的损失函数或者是权值
model.compile(optimizer='rmsprop',loss='binary_crossentropy',
              loss_weights=[1.,0.2])
#fit
model.fit([headline_data,additional_data],[labels,labels],np_epoch=50,batch_size=32)

#因为我们的输入和输出是被命名过的，在定义是传递了name， 我们也可以使用下面的方式编译和训练模型
model.compile(optimizer='rmsprop',
              loss={'main_output':'binary_crossentropy',
                    'aux_output':'binary_crossentropy'},
              losss_weights={'main_output':1,
                             'aux_ouput':0.2})


