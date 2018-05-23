import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=False)

#parameters
learning_rate=0.01
training_epochs=5
batch_size=256
display_step=1
examples_to_show=10

# Network Parameters
n_input = 784

#tf.Graph input
X=tf.placeholder('float',[None,n_input])

# hidden layer settings
n_hidden_1=256
n_hidden_2=128
weights={
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input]))
}
biases={
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_input]))
}

#Builder the encoder
def encoder(x):
    #Encoder Hidden layer with sigmoid activation
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    #eecoder hidden layer with sigmoid activation
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2

#Bulider the decoder

def decoder(x):
    #decoder sigmoid
    layer_1=tf.nn.sigmoid=(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer_2=tf.nn.sigmoid=(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
    return layer_2

#construct mode1
encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

#Prediction
y_pred=decoder_op
#Target (labels) are the input data.
y_true=X

#Define lass and optimizer,minimize the squared error
cost =tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

#launch the graph
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    #training cycle
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)#max(x)=1,min(x)=0
            #run optimization op (backprop) and cost op(to get loss value)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})
            #Display logs per epoch step
        if epoch %display_step==0:
            print("epoch:",'%04d'%(epoch+1),'cost=',"{:.9f}".format(c))

    print("optimization Finished!")

    #applying encode and decoder over test set
    encode_decode=sess.run(y_pred,feed_dict={X:mnist.test.images[:examples_to_show]})
    f,a=plt.subplots(2,10,figsize=(10,2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    plt.show()


