import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs,in_size,out_size,activation_function=None):
    #随机生成一个 矩阵 矩阵！！！第一个参数是shape
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])) + 0.1 #推荐不为0
    Wx_plus_b=tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
#300个 加一个纬度->变成列！向量矩阵了（原来是一维矩阵） 300行！！！shape：[300,1]
x_data=np.linspace(-1,1,300)[:,np.newaxis]
#正态分布
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
#这里输入的大小为什么是1 输出为什么是10 ？？：输入有300个，每次输入是1个数，隐藏层10层。
#hidden_layer:2
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
predition = add_layer(l1,10,1,activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=1))


train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init =tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
#<<<<<<<<<<<<<<可视化>>>>>>>>>>>>>>>>#

fig = plt.figure()#一个图片窗口
ax=fig.add_subplot(1,1,1)#前两个参数把区域划分为：parameter1*para2个区域 在para3上画图

ax.scatter(x_data,y_data)#绘制散点图
plt.ion()
plt.show()#show之后暂停
#.................................... .
for i in range(1000):

    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i% 50==0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value=sess.run(predition,feed_dict={xs:x_data})
        lines=ax.plot(x_data,prediction_value,"r-",lw=5)#red 宽度：5
        #ax.lines.remove(lines[0])#抹除
        plt.pause(0.1)#暂停0.1秒

