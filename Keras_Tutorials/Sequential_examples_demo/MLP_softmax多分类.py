from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD

model=Sequential()
model.add(Dense(64,input_dim=20,init='uniform'))#init:初始化weight
model.add(Activation('tanh'))
model.add(Dropout(0.5))#drop rate
model.add(Dense(10,init='uniform'))
model.add(Activation('softmax'))

sdg=SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sdg,metrics=['accuracy'])
model.fit(X_trian,Y_train,np_epoch=20,batch_size=16)

score=model.evaluate(X_Test,y_test,batch_size=16)

