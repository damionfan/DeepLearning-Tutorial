import numpy as np
import pandas as pd

df=pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
print(df)
#    id raw_grade
# 0   1         a
# 1   2         b
# 2   3         b
# 3   4         a
# 4   5         a
# 5   6         e

'''将原始成绩转换为分类数据类型。'''
df['grade']=df['raw_grade'].astype('category')
print(df)
#    id raw_grade grade
# 0   1         a     a
# 1   2         b     b
# 2   3         b     b
# 3   4         a     a
# 4   5         a     a
# 5   6         e     e
'''将类别重命名'''
df['grade'].cat.categories=['very good','good','very bad']
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
print(df['grade'])
# 0    very good
# 1         good
# 2         good
# 3    very good
# 4    very good
# 5     very bad
# Name: grade, dtype: category
# Categories (5, object): [very bad, bad, medium, good, very good]
'''排序按类别划分的，不是词汇顺序'''
print(df.sort_values(by='grade'))
#    id raw_grade      grade
# 5   6         e   very bad
# 1   2         b       good
# 2   3         b       good
# 0   1         a  very good
# 3   4         a  very good
# 4   5         a  very good
'''按分类列分组还显示空类别。'''
print(df.groupby('grade').size())
# grade              这是空的
# very bad     1
# bad          0
# medium       0
# good         2
# very good    3
# dtype: int64