from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPool2D
from kears.layers import SGD

model=Sequential()
#image is 100*100*3
model.add(Convolution2D(32,kernel_size=3,strides=3,border_moel='valid',input_shape=(3,10,10)))#!!!!!!!!!!!
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3,padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#flatten
model.add(Flatten())
#automatic shape inference
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation('softmax'))

sdg=SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(optimizer=sdg,loss='categorical_crossentropy')

model.fit(X_trian,Y_trian,batch_size=32,np_epoch=1)
