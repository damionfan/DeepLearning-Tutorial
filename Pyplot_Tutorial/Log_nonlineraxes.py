import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

''' 支持线性比例尺和 对数，logit逻辑比例尺 
    改变比例尺：plt.scale('log')
'''

#四个图 ，相同数据 different scales for the y axis

from matplotlib.ticker import NullFormatter

#random
np.random.seed(1)
#make up some data in the interval [0,1]
y=np.random.normal(loc=0.5,scale=0.4,size=1000)#numpy的高斯分布：loc=mean scale=stddev ,size=None(default)
y=y[(y>0)&(y<1)]                              #size : int or tuple of ints, optional output shape. None的话返回一个值
y.sort()
x=np.arange(len(y))
#linear
plt.subplot(221)
plt.plot(x,y)
plt.yscale('linear')#plt.scale()!
plt.title('linear')
plt.grid(True)#网格

#log
plt.subplot(222)
plt.plot(x,y)
plt.yscale('log')#plt.scale()!
plt.title('log')
plt.grid(True)

#symmetric log 对称log
plt.subplot(223)
plt.plot(x,y-y.mean())
plt.yscale('symlog',linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

#logit 逻辑
plt.subplot(224)
plt.plot(x,y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)

#------------important !!!
'''使用NullFormatter 把y轴的次标签变为空字符串，避免使用过多的labels'''
plt.gca().yaxis.set_minor_formatter(NullFormatter())
'''调整subplot 子图布局，因为logit 可能需要更多的空间 因为y-tick label 类似：1-10^(-3) '''
plt.subplots_adjust(top=0.92,bottom=0.08,left=0.1,right=0.95,hspace=0.25,wspace=0.35)
#plt.savefig() 要show（）之前。show()之后就是一个空的了
plt.show()
plt.close('all')