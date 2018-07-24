import tensorflow as tf
'''
tf.summary
visualize our summary statistic during our training 静态的可视化
tf.summary.scalar 纯量
tf.summary.histogram 矩型图  
tf.summary.image
'''
'''step1 create summaries'''
with tf.name_scope('summaries'):
    tf.summary.scalar('loss',self.loss)
    tf.summary.scalar('accuary',self.accuary)
    tf.summary.histogram("histogram loss",self.loss)
    #merge all
    self.summary_op=tf.summary.merge_all()#仅run这一个就可以了

'''step 2: run them'''
loss_batch,_,summary=sess.run([model.loss,model.optimizer,model.summary_op],feed_dict)
#summaries are ops

'''step3:write summaries to file'''
writer=tf.summary.FileWriter(logdir,sess.graph)
writer.add_summary(summary,global_step=step)#填充
#tensorboard --logdir=''