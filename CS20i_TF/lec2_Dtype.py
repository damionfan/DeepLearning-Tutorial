import tensorflow as tf
# TensorFlow take python natives types:boolean ,numeric(int ,float) strings
'''
0-dim tensor or 'scalar;标量
t_0=19
tf.zeros_like(t_0) :0
tf.ones_like(t_0) :1
'''

'''
1-dim tensor,'vector'向量
t_1=['apple','peach','banana']
tf.zeros_like(t_1) :['','','']
tf.ones_like(t_1) :error!
'''

'''
2-dim tensor,'matrix'
tf.zeros_like :all elements are False
tf.ones_like: all elements are True
'''

'''
tf.int32==np.int32
tf.ones([2,2],np.float32)=>[[1.0 1.0],[1.0 1.0]]
for tf.Session.run(fetches) #if the requested fetch is tensor,then the output of will be a Numpy ndarray
'''

# import tensorflow as tf
# my_const =tf.constant([1.0,2.0],name='my_const')
# with tf.Session() as sess:
#     print(sess.graph.as_graph_def())
'''
tf.Variable() ops
X=tf.Variable()
x.initializer #init op ,no ()
x.value()#read op
x.assign()#write op 
x.assign_add()#
'''

'''
a=tf.Variable(2,name="scalar")

b=tf.Variable([2,3],name='vector')

c=tf.Variable([[0,1],[2,3]],name="matrix")
#Variable(initial_value,name) zeros(shape)
W=tf.Variable(tf.zeros([784,10]))
#the easiest way is initializing all varialbes at once全局
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
#initialize only a subset of variables 初始化部分子变量
init_ab =tf.variables_initializer([a,b],name='init_ab')
with tf.Session() as sess:
    sess.run(init_ab)
#初始化单个变量
with tf.Session() as sess:
    sess.run(W.initializer)#没有（） 不是个方法
    print(W.eval())
'''


# W=tf.Variable(10)
# W.assign(100)
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print(W.eval())# 10!
# #W.assign(100) 并没有分配100值给W，它创造了个分配的操作，此操作需要run才能起作用
# #doesn't assign the value 100 to W . it creates an assign op, and that op needs to be run to take effect
# W=tf.Variable(10)
# assign_op=W.assign(100)
# with tf.Session() as sess:
#     #sess.run(W.initializer) assign_op 可以为你初始化，不用这个行了 其实：initialzer使用了assign ！！
#     sess.run(assign_op)
#     print(W.eval())# now is 100


# my_var=tf.Variable(2)
# my_var_times_two=my_var.assign(2*my_var)
# with tf.Session() as sess:
#     sess.run(my_var.initializer)
#     sess.run(my_var_times_two)#4
#     sess.run(my_var_times_two)#8
#     print(my_var.eval())

# my_var=tf.Variable(10)
# with tf.Session() as sess:
#     sess.run(my_var.initializer)#10   ##----assign_add() assign_sub()不能帮你initializer,因为他需要原始的数据
#     sess.run(my_var.assign_add(10))#20
#     sess.run(my_var.assign_sub(10))#back to 10
#

'''---------------Each Session maintain its own copy of variable 每一个session保留自己的copy '''
# W=tf.Variable(10)
# sess1=tf.Session()
# sess2=tf.Session()
#
# sess1.run(W.initializer)
# sess2.run(W.initializer)
#
# print(sess1.run(W.assign_add(10)))
# print(sess2.run(W.assign_sub(10)))
#
# sess1.close()
# sess2.close()

'''-------------use a variable to initialize another variable'''
# #want to declare U=W*2
# W=tf.Variable(tf.truncated_normal([700,10]))
# # U=tf.Variable(2*W)
# U=tf.Variable(2*W.initialized_value())#ensure that W is initalized before its value to initilize U

'''-------------Control Dependencies 依赖关系 程序推进顺序'''
#tf.Graph.control_dependencies(control_inputs)
#define which ops run first
#your graph g have 5 ops :a,b,c,d,e
# with g.control_dependencies([a,b,c]):
#     #'d'and 'e' will only run 'a','b' and 'c' have executed
#     d=...
#     e=...

'''-------Placeholder----------are vaild ops'''
#we,or our clients,can later supply their own data when they need to execute the computation
#tf.placeholder(dtype,shape=None,name=None) shape=None ：任何形状的tensor都可以接受
# create a placeholder of type float 32 -bit ,shape is a vector of 3 elements
# a=tf.placeholder(tf.float32,shape=[3])#vector!
# b=tf.constant([5,5,5],tf.float32)
# c=a+b #short for tf.add(a,b)
# with tf.Session() as sess:    #print(sess.run(c))"""feed value when run"""
#     print(sess.run(c,feed_dict={a:[1,2,3]}))#   feed [1,2,3] to placeholder a via the dict{a:[1,2,3]}
'''feed_dict key是tensor'''

'''tf.Graph.is_feedable(tensor) True if and only if tensor is feedable'''
#不是placeholder 也行
#create operations,tensors,etc(using the default graph)
# a=tf.add(2,5)
# b=tf.multiply(a,3)
# with tf.Session() as sess:
#     #define a dictionary that says to replace the value of 'a' with 15
#     replace_dict={a:15}
#     #run the session,passing in 'replace_dict'as the value to 'feed_dict'
#     sess.run(b,feed_dict=replace_dict)
#
'''--------lazy loding init when you need'''
x=tf.Variable(10,name='x')
y=tf.Variable(20,name='y')
z=tf.add(x,y)#you create the node for add node before executing the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('./graphs',sess.graph)
    for _ in range(10):
        sess.run(z)
    writer.close()
'''tensrorbaord 加载会慢 
解决：1 分离定义和计算，运行操作
     2 use python property to ensure function is also loaded once the first time it is called 
'''
