import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
text() 可以在任意位置添加文本
xlabel() ylabel() title()指定位置
'''

np.random.seed()#默认的话 是真随机 ，设置相同的种子，出现的随机序列相同

mu,sigma=100,15
print(np.random.randn(10000))
x=mu+sigma*np.random.randn(10000)

#数据直方图

n,bins,patches=plt.hist(x,50,normed=1,facecolor='g',alpha=0.75)
'''
所有的text()返回 matplotlib.text.Text 的示例
可以使用关键字传参或者使用setp()
'''
plt.xlabel('Smarts',fontsize=14,color='red')
plt.ylabel('Probability')
plt.title("Histogram of IQ ")
plt.text(60,0.25,r"$\mu=100,\ \sigma=15$")
plt.axis([40,160,0,0.03])
plt.grid(True)
plt.show()
plt.close('all')

''' 在文本中使用数学表达式 用$符号包围'''
plt.title(r'$sigma_i=15$')#r :不转义
''' 注释文本
text() 可以放在轴的任意位置上，annotate()：注释功能 ：1.注释位置xytext 2.文本位置xy 都是（x,y)元组
'''

t=np.arange(0.0,5.0,0.01)
x=np.cos(2*np.pi*t)

line,=plt.plot(t,x,lw=2)
plt.annotate('local max',xy=(2,1),xytext=(3,1.5),arrowprops=dict(facecolor='black',shrink=0.05))
plt.ylim(-2,2)
plt.show()
plt.close()