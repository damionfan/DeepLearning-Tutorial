# coding=utf-8
import keras
from keras.models import Model
from keras.layers import Input,Dense

#return a tensor
inputs=Input(shape=(784,))

x=Dense(64,activation='relu')(inputs)
x=Dense(64,activation='relu')(x)

predictions=Dense(10,activation='softmax')(x)

#this create a model that includes the Input layer and the Dense layers

model=Model(inputs=inputs,output=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data,label)

#所有的模型都是可调用的，like layers
#通过泛型模型的接口，我们可以很容易的重用已train的模型
y=model(x)

#一个对视频分类的demo
from keras.layers import TimeDistributed
input_sequences=Input(shape=(20,784))
processed_sequences=TimeDistributed(model)(input_sequences)

'''--------------------多输入和多输出model--------------------------
see another py file
'''
