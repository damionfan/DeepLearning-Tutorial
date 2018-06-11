import numpy as np
import matplotlib.pyplot as plt

'''
pyplot 有当前figure(图像)和当前axes(轴)的概念。所有的绘图命令都是应用于当前轴。gca()(get_current_axes())获得当前轴 gcf()获得当前figure
'''

# def f(t):
#     return np.exp(-t)*np.cos(2*np.pi*t)

# t1=np.arange(0.,5.,0.1)
# t2=np.arange(0.,5.,0.02)

# plt.figure(1)
# plt.subplot(211)
# plt.plot(t1,f(t1),'bo',t2,f(t2),"k")
#
# plt.subplot(212)
# plt.plot(t2,np.cos(2*np.pi*t2),'r--')
# plt.show()

'''
figure()  可选 ，figure(1) 默认创建 ， 
subplot(111) 也是默认创建 
写成subplot（m,n,p）或者subplot（mnp）。
subplot是将多个图画到一个平面上的工具。其中，m表示是图排成m行，n表示图排成n列，也就是整个figure中有n个图是排成一行的，一共m行，
如果第一个数字是2就是表示2行图。p是指你现在要把曲线画到figure中哪个图上，最后一个如果是1表示是从左到右第一个位置。
'''
#figure()
plt.figure(1)
plt.subplot(211)
plt.plot([1,2,3])
plt.subplot(212)
plt.plot([4,5,6])

plt.figure(2)
plt.plot([4,5,6])

plt.figure(1) #figure 1 is current subplot (211) is current
plt.subplot(211)#211 current
plt.title("easy as 1,2,3")

plt.show()

plt.close('all')
'''
clf() 清空当前figure
cla() 清空当前axes
'''
'''warning::::不实用 plt.close('all')的话，它会一直到占满内存 杀死进程也没用（会保持引用）'''