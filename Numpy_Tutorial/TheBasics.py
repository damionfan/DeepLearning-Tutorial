import tensorflow as tf
import numpy as np

'''-----------------------------基础知识----------------------------
在numpy中 维度成为轴 
[1,2,3] 一个轴，长度为3
[[1,2,3],
[4,5,6]] 两个轴 ，第一个轴长度为2 ，第二轴长度为3 ，轴从高纬->到低纬
数组类为ndarry 即使numpy.array也和python 的array不同，前者处理的多


ndarry.ndim :维度数/轴数,有多少个轴
ndarry.shape:(n,m) 元组 
ndarry.size : 元素总数量 =n*m
ndarry.dtype : np.int32,nnp.int16等
ndarry.itemsize : 每个元素的大小（bytes）float64: 64/8=8 ,和ndarry.dtype.itemsize相等
ndarry.data : 缓冲区 ，含有数组实际元素 一般不用
'''
a=np.arange(15).reshape(3,5)
# print(a,a.shape,a.ndim,a.dtype.name,a.itemsize,a.size,type(a))
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]] (3, 5) 2 int64 8 15 <class 'numpy.ndarray'>

'''--------------------数组创建---------------------'''
a=np.array([1,2,3,4])#dtype=int64
b=np.array([1.2,2.4])#dtype=float64
'''但是使用多个数字不作为数组：
a=np.array([1,2,3]) 对
a=np.array(1,2,3) 错 '''
b=np.array([(1,3),(4,5,6)])
# print(b) [(1, 3) (4, 5, 6)]
'''可以在创建的过程中显示定义数组类型'''
c=np.array([[1,2],[3,4]],dtype=complex)
# print(c)
# [[1.+0.j 2.+0.j]
#  [3.+0.j 4.+0.j]]
'''zeros：充满0的数组，ones：充满1的数组，empty：内容随机 取决内存状态。默认dtype:int64'''
a=np.zeros((3,4))
b=np.ones((2,3,4),dtype=innt16)
c=np.empty((2,3))