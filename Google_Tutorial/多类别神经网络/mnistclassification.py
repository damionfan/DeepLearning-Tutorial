import glob,math,os
from matplotlib import cm,gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.1f}'.format

mnist_dataframe=pd.read_csv('mnist_train_small.csv',sep=',',header=None)
mnist_dataframe=mnist_dataframe.head(10000)
mnist_dataframe=mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
# print(mnist_dataframe.head(3)) 784+1 cols
def parse_labels_and_features(dataset):
    labels=dataset[0]#列
    features=dataset.loc[:,1:784]
    features/=255
    return labels,features

training_targets,training_examples=parse_labels_and_features(mnist_dataframe[:7500])#切片
validation_targets,validation_examples=parse_labels_and_features(mnist_dataframe[7500:])

#show example and label
def show_one_image():
    rand_example=np.random.choice(training_examples.index)
    _,ax=plt.subplots()#return figure an ax
    ax.matshow(training_examples.loc[rand_example].values.reshape(28,28))#loc[index] 一行#values没有列名仅数值
    ax.set_title('label:%i'%training_targets.loc[rand_example])
    ax.grid(False)
    plt.show()
'''--------------------------使用线性模型LinearClassifier'''
'''会使用 log_loss 函数跟踪模型的错误。不应将此函数与用于训练的 LinearClassifier 内部损失函数相混淆。'''
def construct_feature_columns():
    return set([tf.feature_column.numeric_column('pixels',shape=784)])
# ，我们会对训练和预测使用单独的输入函数，并将这些函数分别嵌套在 create_training_input_fn() 和 create_predict_input_fn() 中，
# 这样一来，我们就可以调用这些函数，以返回相应的 _input_fn，并将其传递到 .train() 和 .predict() 调用。
def create_training_input_fn(features,labels,batch_size,num_epochs=None,shuffle=True):
    def _input_fn(num_epochs=None,shuffle=True):
        idx=np.random.permutation(features.index)
        raw_features={'pixels':features.reindex(idx)}
        raw_targets=np.array(labels[idx])
        ds=Dataset.from_tensor_slices((raw_features,raw_targets))
        ds=ds.batch(batch_size).repeat(num_epochs)
        if shuffle:
            ds=ds.shuffle(10000)
        feature_batch,label_batch=ds.make_one_shot_iterator().get_next()
        return feature_batch,label_batch
    return _input_fn
def create_predict_input_fn(features,labels,batch_size):
    def _input_fn():
        raw_features={'pixels':features.values}
        raw_targets=np.array(labels)
        ds=Dataset.from_tensor_slices((raw_features,raw_targets))
        ds=ds.batch(batch_size)
        feature_batch,label_batch=ds.make_one_shot_iterator().get_next()
        return feature_batch,label_batch
    return _input_fn
def train_linearclassifier_model(learning_rate,steps,batch_size,
                                 training_examples,training_targets,validation_examples,validation_targets):
    periods=10
    steps_per_period = steps / periods
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(),
        n_classes=10,
        optimizer=my_optimizer,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1)
    )#检查点保存格式。默认是5

    print('train')
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        #predict
        t_predictions=list(classifier.predict(input_fn=predict_training_input_fn))
        tr_probabilities=np.array([item['probabilities'] for item in t_predictions])
        train_pred_class_id=np.array([item['class_ids'][0] for item in t_predictions])
        training_pred_one_hot=tf.keras.utils.to_categorical(train_pred_class_id,10)

        validation_predictions=list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities=np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id=np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot=tf.keras.utils.to_categorical(validation_pred_class_id,10)

        #loss
        training_log_loss=metrics.log_loss(training_targets,training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    #删除事件文件节省硬盘空间
    _=map(os.remove,glob.glob(os.path.join(classifier.model_dir,'events.out.tfevents*')))#map(function, iterable, ...)
    # 计算最终的准确度
    final_predictions=classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions=np.array([item['class_ids'][0] for item in final_predictions])

    accuracy=metrics.accuracy_score(validation_targets,final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    #混淆矩阵
    cm=metrics.confusion_matrix(validation_targets,final_predictions)
    #normalize the confusion matrix by row
    cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    ax=sns.heatmap(cm_normalized,cmap='bone_r')
    ax.set_aspect(1)
    plt.title('confusion matrix')
    plt.ylabel("true label")
    plt.xlabel('predicted label')
    plt.show()

    return classifier

def train_nn_model(learning_rate,steps,batch_size,hidden_units,
                                 training_examples,training_targets,validation_examples,validation_targets):
    periods = 10
    # Caution: input pipelines are reset with each call to train.
    # If the number of steps is small, your model may never see most of the data.
    # So with multiple `.train` calls like this you may want to control the length
    # of training with num_epochs passed to the input_fn. Or, you can do a really-big shuffle,
    # or since it's in-memory data, shuffle all the data in the `input_fn`.
    steps_per_period = steps / periods
    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column('pixels', shape=784)]

    # Create a DNNClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier

# classifier = train_linearclassifier_model(
#              learning_rate=0.02,
#              steps=100,
#              batch_size=10,
#              training_examples=training_examples,
#              training_targets=training_targets,
#              validation_examples=validation_examples,
#              validation_targets=validation_targets)

classifier = train_nn_model(
    learning_rate=0.05,
    steps=100,
    batch_size=30,
    hidden_units=[100, 100],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

'''test'''
mnist_test_dataframe=pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/mnist_test.csv",
  sep=",",
  header=None)
test_targets,test_examples=parse_labels_and_features(mnist_test_dataframe)
predict_test_input_fn=create_predict_input_fn(test_examples,test_targets,100)
test_predictions=classifier.predict(predict_test_input_fn)
test_predictions=np.array([item['class_ids'][0] for item in test_predictions])
accuracy=metrics.accuracy_score(test_predictions,test_targets)
print('acc:%0.2f'%accuracy)

'''可视化第一个hiddenlayer权重模型的输入层有 784 个权重，对应于 28×28 像素输入图片。
第一个隐藏层将有 784×N 个权重，其中 N 指的是该层中的节点数。
我们可以将这些权重重新变回 28×28 像素的图片，具体方法是将 N 个 1×784 权重数组变形为 N 个 28×28 大小数组。'''
print(classifier.get_variable_names())
# ['dnn/hiddenlayer_0/bias', 'dnn/hiddenlayer_0/bias/t_0/Adagrad', 'dnn/hiddenlayer_0/kernel', 'dnn/hiddenlayer_0/kernel/t_0/Adagrad', 'dnn/hiddenlayer_1/bias', 'dnn/hiddenlayer_1/bias/t_0/Adagrad', 'dnn/hiddenlayer_1/kernel', 'dnn/hiddenlayer_1/kernel/t_0/Adagrad', 'dnn/logits/bias', 'dnn/logits/bias/t_0/Adagrad', 'dnn/logits/kernel', 'dnn/logits/kernel/t_0/Adagrad', 'global_step']
weights0=classifier.get_variable_value('dnn/hiddenlayer_0/kernel')
print('weight0 :shape',weights0.shape)
num_nodes=weight0.shape[1]
num_rows=int(math.ceil(num_nodes/10.0))#ceil想上取整
fig.axs=plt.subplots(num_rows,10,figsize=(20,2*num_rows))
for coef,ax in zip(weights0.T,axes.ravel()):#ravel(): Return a contiguous flattened array.
    #weights in coef is reshaped from 1*784 to 28*28
    ax.matshow(coef.reshape(28,28),cmp=plt.cm.pink)
    ax.set_xticks(())#使用刻度列表设置x 刻度
    ax.set_yticks(())
plt.show()


