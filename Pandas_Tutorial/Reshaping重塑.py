import numpy as np
import pandas as pd
np.random.seed(0)
'''这一章并没有看懂http://pandas.pydata.org/pandas-docs/stable/10min.html'''
'''----------------------stack堆栈--------------------------'''
'''zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表
在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。
http://www.runoob.com/python/python-func-zip.html'''

tuples=list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],
                  ['one', 'two', 'one', 'two','one', 'two', 'one', 'two']]))
# [('bar', 'one'), ('bar', 'two'), ('baz', 'one'), ('baz', 'two'), ('foo', 'one'), ('foo', 'two'), ('qux', 'one'), ('qux', 'two')]
index=pd.MultiIndex.from_tuples(tuples,names=['first','second'])
df=pd.DataFrame(np.random.randn(8,2),index=index,columns=['A','B'])
df2=df[:4]
print(df2)
'''使用stack()方法压缩DataFrame列的级别'''
stacked=df2.stack()
print(stacked)
# first  second
# bar    one     A    1.764052
#                B    0.400157
#        two     A    0.978738
#                B    2.240893
# baz    one     A    1.867558
#                B   -0.977278
#        two     A    0.950088
#                B   -0.151357
# dtype: float64
