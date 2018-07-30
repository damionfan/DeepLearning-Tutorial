import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
max_steps=1000
lr=1e-3
dropout=0.9
data_dir=''
log_dir='/log'
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y_=tf.placeholder(tf.float32,[None,10],name='y-input')

with tf.name_scope('input_reshape'):
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)#max_image=10

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))
'''计算variable的mean，stddev,max,min然后使用tf.summary.scalar'''
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min'.tf.reduce_min(var))
        tf.summary.histogram('histogram',var)


def nn_layer(input_tensor,input_dim,out_dim,layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights=weight_variable([input_dim,output_dim])
        with tf.name_scope('biases'):
            biases=bias_variable([output_dim])
        with tf.name_scope('wx_plus_b'):
            preactivate = tf.matmul(input_tensor,weights)+bias
            tf.summary.histogram('pre_activations',preactivate)
        activations=act(preactivate,name='activation')
        tf.summary.histogram('activations',activations)
        return activations
hidden1=nn_layer(x,784,500,'layer1')

with tf.name_scope("dropout"):
    keep_prob=tf.plaeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability',keep_prob)
    dropped=tf.nn.dropout(hidden1,keep_prob)

y=nn_layer(dropped,500,10,'layer2',act=tf.identity)

with tf.name_scope('cross_entropy'):
    diff=tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
    with tf.name_scope('total'):
        cross_entropy=tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(lr).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)

merged=tf.summary.merge_all()

with tf.Session() as sess:
    train_writer=tf.summary.FileWriter(log_dir+'/train',sess.graph)
    test_writer = tf.summary.FileWriter(log_dir+'test')
    sess.run(tf.global_variables_initializer())
    def feed_dict(train):
        if train:
            xs,ys=mnist.train.next_batch(100)
            k=dropout
        else:
            xs,ys=mnist.test.image,mnist.test.labels
            k=1.0
        return {x:xs,y_:ys,keep_prob:k}
    saver=tf.train.Saver()
    for i in range(max_steps):
        if i%10 ==0:
            summary,acc=sess.run([merged,accuracy],feed_dict=feed_dict(False))
            test_writer.add_summary(summary,i)
            print('accuracy at step %s :%s '%(i,acc))
        else :
            if i %100 ==99:
                run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata=tf.RunMetadata()
                summary,_=sess.run([merged,train_step],feed_dict=feed_dict(True),
                                   options=run_options,run_metadata=run_metadata)
                train_writer.add_run_metadat(run_metadata,'step%03d'%i)
                train_writer.add_summary(summary,i)
                saver.save(sess,logdir+'/model.ckpt',i)
                print("addinng run metadata for ",i)
            else:
                summary,_=sess.run([merged,train_step],feed_dict=feed_dict(True))
                train_writer.add_summay(summary,i)



