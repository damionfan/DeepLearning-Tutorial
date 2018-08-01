import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
大致了解 pandas 库的 DataFrame 和 Series 数据结构
存取和处理 DataFrame 和 Series 中的数据
将 CSV 数据导入 pandas 库的 DataFrame
对 DataFrame 重建索引来随机打乱数据
DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
Series，它是单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。'''

city_name=pd.Series(['sanFracisco','san jose','sacramento'])
population=pd.Series([8524,10102,23509])
cities=pd.DataFrame({'city name':city_name,'population':population})

'''下面的示例加载了一个包含加利福尼亚州住房数据的文件。请运行以下单元格以加载数据，并创建特征定义：'''
california_housing_dataframe=pd.read_csv('california_housing_train.csv',sep=',')
print(california_housing_dataframe.describe())
print(california_housing_dataframe.head())
# california_housing_dataframe.hist('housing_median_age')#画histogram
# plt.show()
'''---------------操控数据---------------------'''
np.log(population)
print(population.apply(lambda x:x>1000000))
# 0    False
# 1    False
# 2    False
# dtype: bool
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['population'] / cities['Area square miles']
print(cities)
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['city name'].apply(lambda name: name.startswith('San'))
print(cities.index)#RangeIndex(start=0, stop=3, step=1)
'''DataFrame.reindex 以手动重新排列各行的顺序。'''
'''重建索引是一种随机排列 DataFrame 的绝佳方式。我们会取用类似数组的索引，
然后将其传递至 NumPy 的 random.permutation 函数，
该函数会随机排列其值的位置。如果使用此重新随机排列的数组调用 reindex，
会导致 DataFrame 行以同样的方式随机排列。'''
'''索引和行一直是对应的！！'''
cities.reindex(np.random.permutation(cities.index))#根据index排序 permutation: 排列
print(cities)
'''reindex 会为此类“丢失的”索引添加新行，并在所有对应列中填充 NaN 值：'''
print('-----------------------------------')
#
# for key ,value in dict(cities).items():
#     print('key : %s, value :%s'%(key,value))
print(california_housing_dataframe[['total_rooms']])