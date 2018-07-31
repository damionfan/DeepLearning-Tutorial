import tensorflow as tf

#建立一个线性分类器
classifier  = tf.estimator.LinearClassifier()

#在数据集上训练一个model
classifier.train(input_fn=train_input_fn,steps=2000)
#steps:迭代次数，一步一次计算loss，然后修改
#batch_size ：单步的data数
#总训练的数据量=steps*batch_size

# 预测
predictions=classifier.predict(input_fn=predict_input_fn)

