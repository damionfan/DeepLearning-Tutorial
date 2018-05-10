'''
a library to load the mnist image data. for details of the data structures that are returned ,
see the doc string for ''load_data"" and "load_data_wrapper".in practice,'load_data_wrapper' is the function
usually called by our neural network code
'''
#standard library
#import cPickle -python2
import pickle
import gzip
#thrid-party library
import numpy as np

def load_data():
    f=gzip.open('./data/mnist.pkl.gz','rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
    f.close()
    return (training_data,validation_data,test_data)

def load_data_wrapper():
    tr_d,va_d,te_d=load_data()
    training_inputs=[np.reshape(x,(784,1))for x in tr_d[0]]
    training_results=[vectorized_result(y) for y in tr_d[1]]
    training_data=zip(training_inputs,training_results)
    validation_inputs=[np.reshape(x,(784,1))for x in va_d[0]]
    validation_data=zip(validation_inputs,va_d[1])
    test_inputs=[np.reshape(x,(784,1)) for x in te_d[0]]
    test_data=zip(test_inputs,te_d[1])
    return (training_data,validation_data,test_data)

def vectorized_result(j):
    e=np.zeros((10,1))
    e[j]=1.0
    return e