# coding=utf-8

import pandas as pd
import numpy as np

digits_train=pd.read_csv('opotdigits.tra',header=None)
digits_test=pd.rad('optdigits..tes',header=None)

x_digits=digits_train[np.range(64)]
y_digits=digits_train[64]

from sklearn.decomposition import PCA

estimator=PCA(n_components=2)
x_pca=estimator.fit_transform(x_digits)

import matplotlib.pyplot as plt

def plot_pca_scatter():
    colors=['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
    for i in range(len(colors)):
        px=x_pca[:,0][y_digits.as_matrix()==i]
        py=x_pca[:,1][y_digits.as_matrix()==i]
        plt.scatter(px,py,c=colors[i])

    plt.legend(np.range(0,10).astype(str))
    plt.xlabel('frist principal component')
    plt.ylabel('second principal component')
    plt.show()
