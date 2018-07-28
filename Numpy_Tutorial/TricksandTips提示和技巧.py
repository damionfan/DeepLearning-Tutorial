import numpy as np
import matplotlib.pyplot as plt
'''--------------自动整型-----------------------'''
a=np.arange(30)
a.shape=2,-1,3#-1：自动计算
print(a.shape)
# (2, 5, 3)
'''--------------矢量堆叠-----------------------'''
x=np.arange(0,10,2)#[0 2 4 6 8]
y=np.arange(5)#[0 1 2 3 4]
m=np.vstack([x,y])
# [[0 2 4 6 8]
#  [0 1 2 3 4]]
xy=np.hstack([x,y])#[0 2 4 6 8 0 1 2 3 4]

'''-----------------直方图-----------------------'''

mean,scale=2,0.5#scale=stddiv
'''loc：float
    此概率分布的均值（对应着整个分布的中心centre）
scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints
    输出的shape，默认为None，只输出一个值'''
x=np.random.normal(mean,scale,size=10000)
'''
x : (n,) array or sequence of (n,) arrays
这个参数是指定每个bin(箱子)分布的数据,对应x轴
bins : integer or array_like, optional
这个参数指定bin(箱子)的个数,也就是总共有几条条状图
normed : boolean, optional
If True, the first element of the return tuple will be the counts normalized to form a probability density, i.e.,n/(len(x)`dbin)
这个参数指定密度,也就是每个条状图的占比例比,默认为1
color : color or array_like of colors or None, optional
这个指定条状图的颜色'''
plt.hist(v,bins=50,normed=1)
plt.show()
(n,bins)=np.histogram(x,bins=50,normed=True)#仅生成数据
plt.plot(.5*(bins[1:]+bins[:-1]),n)
plt.show()