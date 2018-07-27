import numpy as np

'''--------------------全部没有复制-------------------'''
'''没有新的对象创建，b只是a的一个别名
   简单的分配不会复制数组对象或其数据'''
a=np.arange(12)
b=a
b.shape=3,4
print(a is b)#True
'''函数调用不会复制'''
def f(x):
    return (id(x))
print(id(a))#2856185284928
print(f(a))#2856185284928
'''------------------view和浅拷贝----------------------'''
'''不同的数组对象可以共享相同的数据 view()方法创建一个相同数据的新对象'''
c=a.view()
print(c is a )#False
print(c.base is a)#True
c.shape=2,6
'''a的shape没变'''
print(a.shape,c.shape)#(3,4) (2, 6)
'''a的data变了！'''
c[0,4]=1234
print(a)
# [[   0,    1,    2,    3],
#  [1234,    5,    6,    7],
#  [   8,    9,   10,   11]]
'''切片也会返回一个view'''
'''-------------------深拷贝----------------------------
copy（）'''
d=a.copy()
print(d is a)#False
print(d.base is a)#False
d[0,0]=9999#对A没影响



