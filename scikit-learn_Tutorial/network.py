from sklearn import datasets,svm,random_projection
from sklearn.externals import joblib
import pickle
import numpy as np
np.random.seed(0)
'''加载数据'''
iris=datasets.load_iris()
digits=datasets.load_digits()
# print(digits.data)
# print(digits.target)
'''学习和预测'''
clf=svm.SVC(gamma=0.001,C=100)#分类器
'''fit(x,y) predict(T)'''
clf.fit(digits.data[:-1],digits.target[:-1])#最后一个没用
print(clf.predict(digits.data[-1:]))
'''模型持久性pickle'''
s=pickle.dumps(clf)
clf2=pickle.loads(s)
print(clf2.predict(digits.data[-1:]))
'''使用joblib替代pickle（joblib.dump&joblib.load)'''
joblib.dump(clf,'clf.pkl')
clf3=joblib.load('clf.pkl')
'''约定'''
'''类型转换，输入转换为float64'''
rng=np.random.RandomState(0)
X=rng.rand(10,2000)
X=np.array(X,dtype='float32')
print(X.dtype)
transformer=random_projection.GaussianRandomProjection()
X_new=transformer.fit_transform(X)
print(X_new.dtype)#float64

clfs=svm.SVC()
clfs.fit(iris.data,iris.target)
print(list(clfs.predict(iris.data[:3])))
clfs.fit(iris.data,iris.target_names[iris.target])
print(list(clfs.predict(iris.data[:3])))
'''改进和更新参数'''
clf.set_params(kernel='linear').fit(X,y)
clf.predict(X_test)
clf.set_params(kernel='rbf').fit(X,y)
clf.predict(X_test)
'''多类和多标签拟合'''
from sklearn.multiclass import oneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif=OneVsRestClassifier(estimator=svm.SVC(random_state=0))
classif.fit(X,y).predict(x)

from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y=MultiLabelBinarizer().fit_transform(y)
classif.fit(X,y).predict(X)


