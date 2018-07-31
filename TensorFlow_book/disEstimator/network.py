import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from sklearn import datasets,cross_validation
'''（feature,target)->(predictions,loss,train_op)'''
def my_model(features,target):
    target=tf.one_hot(target,3,1,0)
    #3- the depth of the one hot dimension.
#  [[1., 0., 0.],#长度是3！因为输入数据的种类和长度以一样的，索引一般这depth也可以用class理解
#  [0., 1., 0.],
#  [0., 0., 1.]]
    features=layers.stack(features,layers.fully_connected,[10,20,10])#叠加多个fc层，每层节点10,20,10
    prediction,loss=tf.contrib.learn.models.logistic_regression_zero_init(features,target)#初始0的逻辑回归
    train_op=tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),optimizer='Adagrad',learning_rate=0.1)
    return {'class':tf.argmax(prediction,1),'prob':prediction},loss,train_op

iris=datasets.load_iris()
x_train,x_test,y_train,y_test=cross_validation.train_test_split(iris.data,iris.target,test_size=0.2,random_state=35)

classifier=learn.Estimator(model_fn=my_model)
classifier.fit(x_train,y_train,steps=700)

predictions=classifier.predict(x_test)
