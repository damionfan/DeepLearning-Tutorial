import matplotlib.pyplot as plt
import numpy as np
'''
                    plt.plot:
接受任意数量参数 
仅一组数据的话，默认是给y轴。然后自动生成x轴（同长度的，从0开始)
有一个可选的第三个参数：串联表示颜色和线性 默认'b-':蓝色实线
'''
#plt.plot([1,2,3,4])#仅一组数据的话，默认是给y轴。然后自动生成x轴（同长度的，从0开始)
# plt.plot([1,2,3,4],[1,4,9,16],'ro')#x,y,红色 圆点
'''
plt.axis[xmin,xmax,ymin,ymax]
输入序列在内部转换为numpy
'''
# plt.axis([0,6,0,20])
# plt.ylabel('some numbers')
# plt.show()

'''详细示例'''
#每0.2取样
t=np.arange(0.,5.,0.2)#numpy.arange([start, ]stop, [step, ]dtype=None) start:默认0 step：默认1.  5.=5.0

#red bashes,blue squares ,green triangles
#红色破折号，蓝色方块，绿色三角
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()

'''plot 的各种属性：linewidth(线宽),dash style(短划线风格),antialiased (抗锯齿) '''
plt.plot(x,y,linewidth=2.0)


'''line2d line1,line2=plt.plot(x2,y1,x2,y2)'''#----------返回两个！！！
line,=plt.plot(x,y,'-')
line.set_antialiased(False)

'''使用setp()函数 设置属性'''
lines=plt.plots(x1,y1,x2,y2)#---------这是两个！
plt.setp(lines,color='r',linewidth=2.0)
plt.setp(lines,'color','r','linewidth',2.0)#matlab style
#使用多行来座位参数
lines=plt.plot([1,2,3])
plt.setp(lines)
...

plt.close('all')

