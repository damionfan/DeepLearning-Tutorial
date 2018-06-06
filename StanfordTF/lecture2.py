import tensorflow as tf
# a=tf.constant(2,name='a')
# b=tf.constant(3,name='b')
# x=tf.add(a,b,name='add')
#
# with tf.Session() as sess:
#     #add this line to use TensorBoard
'''     writer=tf.summary.FileWriter('./graphs',sess.graph)'''
#     print(sess.run(x))
# writer.close()#close the writer!!! dont frogot it !

# '''
# tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)
# '''
# a=tf.constant(2,shape=[2,2])
# tf.InteractiveSession()
# a.eval()
# #[[2,2],[2,2]]
# b=tf.constant([2,1],shape=[3,3])
# b.eval()
# #[[2,1,1],
# # [1,1,1],
# # [1,1,1]]
#
# a=tf.constant([2,2],name='a')
# b=tf.constant([[0,1],[2,3]],name='b')
# x=tf.add(a,b,name='add')
# y=tf.multiply(a,b,name='mul')
# with tf.Session() as sess:
#     sess.run(x)
#     print(x)
'''
#-------------zeros
tf.zeros(shape,dtype=tf.float32,name=None)
creates a tensor of shape and all elements will be zeros (when ran in session)
tf.zeros([2,3],tf.int32)=>[[0,0,0],[0,0,0]] matrix
'''

'''
-----------zeros_like
tf.zeros_like(input_tensor,dtype=None,name=None,optimize=True
creates a tensor of shape and type (unless type is specified) as the input_tensor but all elements are zeros
input_tensor is [[0,1],[2,3],[4,5]]
tf.zeros_like(input_tensor) =>same shape but all elements are zeros
'''
#same :tf.ones tf.ones_like


'''
tf.fill(dims,value,name=None)
creates a tensor filled with a scalar value
tf.fill([2,3],8)=>[[8,8,8],[8,8,8]]
numpy:
create a numpy array a 
a.fill(value)
'''

'''
tf.linspace(start,stop,num,name=None)#slights different from np.linspace
tf.linspace(10.0,13.0,4)=>[10.0,11.0,12.0,13.0]
tf.range(start,limit=None,delta=1,dtype=None,name='range')
#'start' is 3,'limit' is 18,'delta'is 3
tf.range(start,limit,delta)=>[3,6,9,12,15]
#limit=5
tf.range(5)=>[0,1,2,3,4]#没有5！！！ limit no end!!!
Tensor objects are not iterable!!!!!!!!!!!!!!
'''

'''
tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)正态 标准差=1
tf.truncated_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)#截断正态 比上面更加密集 
#产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，

tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)产生均匀分布
tf.random_shuffle(value,seed=None,name=None)#沿着要被洗牌的张量的第一个维度，随机打乱。
#[[1, 2],       [[5, 6],  
 [3, 4],  ==>   [1, 2],  
 [5, 6]]        [3, 4]]  
 
tf.random_crop(value,size,seed=None,name=None) 裁剪 常用用于图片
tf.multinomial(logits,num_samples,seed=None,name=None)
tf.random_gamma(shape,alpha,beta=None,dtype=tf.float32,seed=None,name=None)伽马分布
'''
'''
tf.set_random_seed(seed)#使可以重复
'''
'''
a=tf.constant([3,6])
b=tf.constant([2,2])
tf.add(a,b) #[5,8]
tf.add_n([a,b,b]) ->a+b+b
tf.multiply(a,b) [6,12]
tf.matmul(a,b) !error
tf.matmul(tf.reshape(a,[1,2]),tf.reshape(b,[2,1]) #[[18]] !!!!
tf.div(a,b) [1,3]
tf.mod(a,b) [1,0]
'''