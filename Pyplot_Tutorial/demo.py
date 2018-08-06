import matplotlib.pyplot as plt
import numpy as np
import time
from math import *


for i in range(10):
    y=np.random.rand(100)
    plt.plot(y)
    plt.show()
#我也是服了，不说好了是堵塞模式吗？怎么还是可以一直输出
print('done')