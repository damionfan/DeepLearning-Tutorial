from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
model=Sequential()

model.add(Dense(output_dim=64,input_dim=1000))
model.add(Activation('relu'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accruacy'])
#编译模型必须制定损失函数和优化器，也可以自己定义损失函数

model.fit(X_train,Y_train,nb_epoch=5,batch_size=32)

# model.train_on_batch(X_batch,Y_batch)
#评估
loss_and_metrics=model.evaluate(X_test,Y_test,batch_size=32)
#预测
classes=model.predict_classes(X_test,batch_size=32)
proba=model.predict_proba(X_test,batch_size=32)


