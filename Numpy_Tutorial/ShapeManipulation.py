import numpy as np
from numpy import newaxis
'''-------------改变数组的形状--------------'''
a=np.floor(10*np.random.random((3,4)))#向下取整
# [[ 2.,  8.,  0.,  6.],
#        [ 4.,  5.,  1.,  1.],
#        [ 8.,  9.,  3.,  6.]]
a.shape #(3,4)
'''以下三个命令都返回已修改的数组，但不更改原始数组：'''
a.ravel()#flatten !!!!
#[ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.]
a.T #转置
# [[ 2.,  4.,  8.],
#        [ 8.,  5.,  9.],
#        [ 0.,  1.,  3.],
#        [ 6.,  1.,  6.]]
a.reshape(6,2)
# [[ 2.,  8.],
#        [ 0.,  6.],
#        [ 4.,  5.],
#        [ 1.,  1.],
#        [ 8.,  9.],
#        [ 3.,  6.]]
'''resize 修改数组本身'''
a.resize((2,6))
a
#[[ 2.,  8.,  0.,  6.,  4.,  5.],
# [ 1.,  1.,  8.,  9.,  3.,  6.]]
'''在整型中将尺寸设为-1，会自动计算其他尺寸'''
a.reshape(3,-1)
# [[ 2.,  8.,  0.,  6.],
#        [ 4.,  5.,  1.,  1.],
#        [ 8.,  9.,  3.,  6.]]
'''--------------------折叠不同的数组-----------------'''
a=np.floor(10*np.random.random((2,2)))
a
# [[ 8.,  8.],
#  [ 0.,  0.]]
b=np.floor(10*np.random.random((2,2)))
b
# [[ 1.,  8.],
#  [ 0.,  4.]]
print(np.vstack((a,b)))
'''参数是tuple ,它是垂直（按照行顺序）的把数组给堆叠起来。
沿着第一个维度连接，一维的数组reshape为（1，N)
https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.vstack.html'''
# [[ 8.,  8.],
#  [ 0.,  0.],
#  [ 1.,  8.],
#  [ 0.,  4.]]
print(np.hstack((a,b)))
'''沿着第二维度连接,但是1维的数组沿着第一维
https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.hstack.html?highlight=hstack#numpy.hstack'''
# [[ 8.,  8.,  1.,  8.],
#  [ 0.,  0.,  0.,  4.]]
'''column_stack将一维D数组对应的列 作为 列(元素） 折叠到2-D数组中'''
a=np.array([4,2])
b=np.array([3,8])
np.column_stack((a,b))
# [[4 3]
#  [2 8]] 注意看顺序
np.hstack((a,b))
# [ 4., 2., 3., 8.] hstack对1——D沿着第一D
print(a[:,newaxis])
# [[4]
#  [2]]
np.column_stack((a[:,newaxis],b[:,newaxis]))
# [[ 4.,  3.],
#  [ 2.,  8.]]
'''和hstack一个作用'''
np.hstack((a[:,newaxis],b[:,newaxis]))
# [[ 4.,  3.],
#  [ 2.,  8.]]
'''函数row_stack等效vstack 于任何输入数组'''
'''concatenate 可以指定连接的轴'''
'''np.r_ np.c_类似vstack,hstack'''

'''---------------将一个数组拆分成几个较小的数组----------------'''
a=np.floor(10*np.random.random((2,12)))
print(a)
# [[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
#  [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]]
print(np.hsplit(a,3))
# [array([[ 9.,  5.,  6.,  3.],
#        [ 1.,  4.,  9.,  2.]]), array([[ 6.,  8.,  0.,  7.],
#        [ 2.,  1.,  0.,  6.]]), array([[ 9.,  7.,  2.,  7.],
#        [ 2.,  2.,  4.,  0.]])]
print(np.hsplit(a,(3,4)))
# [array([[ 9.,  5.,  6.],
#        [ 1.,  4.,  9.]]), array([[ 3.],
#        [ 2.]]), array([[ 6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
#        [ 2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])]