import tensorflow as tf
'''
tf.train.Saver
saves graph's variables in binary files 把训练的变量保存在二进制文件中
not save the model 并没有存储模型
#保存sess 而不是graph #第三个参数将训练的次数作为后缀加入到模型名字中。
tf.train.Saver.save(sess,save_path,global_step=None...)

max_to_keep:保持最新的
keep_checkpoint_every_n_hours 自己去查
'''
'''
save parameters after 1000 steps
'''
#define model

#create a saver object
#saver=tf.train.Saver(...variables...)#几个变量
saver=tf.train.Saver({'v1':v1,'v2':v2})#使用dict

#launch a session to compute the graph
with tf.Session() as sess:
    #actual training loop
    for step in range(train_steps):
        sess.run([optimizer])

        if (step+1) % 1000 ==0:
            saver.save(sess,'checkpoint_directory/model_name',global_step=model.global_step)

'''
Global step

self.global_step=tf.Variable(0,dtype=tf.float32,trainabl=False,name="global_step")
self.optimizer=tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,global_step=self.global_step)#告诉optimizer提高global step global_step 会在变量更新后增加1

'''

'''
Restore variables
saver.restore(sess,'checkpoints/name_of_the_checkpoint')

'''

ckpt=tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess,ckpt.model_checkpoint_path)
#checkpoint keeps track of the latest checkpoint
#safeguard to restore checkpoints only when there are checkpoints

sess=tf.Session()
new_saver=tf.train.import_meta_graph('my_model.meta')#saver.save(sess,'my_model')
new_saver.restore(sess,tf.train.latest_checkpoint('./'))#latest_checkpoint:前缀获取。
all_vars=tf.get_collect('vars')#tf.add_to_collection('vars',w1) 依然是tensor
for v in all_vars:
    print(sess.run(v))