from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Embedding
from keras.layers import LSTM

model=Sequential()
model.add(Embedding(max_feature,256,input_length=maxlen))
model.add(LSTM(ouput_dim=128,activation='sigmoid',inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=128,nb_epoch=10)