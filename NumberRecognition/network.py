import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes=sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
        #sizes是一个包含各层神经元个数的列表
        # #第一个向量是3个元素的列向量 第二个是1个元素列向量
        #第一个是3*2 第二个是1*3行
        #zip() 函数把对象对应的元素打包成元祖，然后元组组成的列表，长度与最短的对象相同
         #   random.randn(d0,,,dn)产生n维矩阵

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a=sigmoid(np.dot(w, a)+b)
        return a
        #dot()返回的是两个数组的点积(dot product)

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):

        training_data=list(training_data)
        test_data=list(test_data)
        if test_data:
            #Python3中，zip()函数实现为迭代器，可以随着迭代返回一系列值
            n_test = len(test_data)

        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batchs=[training_data[k:k+mini_batch_size]
                         for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch{0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch{0} complete".format(j))

    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            self.weights=[w-(eta/len(mini_batch))*nw
                          for w,nw in zip(self.weights,nabla_w)]
            self.biases=[b-(eta/len(mini_batch))*nb
                         for b,nb in zip(self.biases,nabla_b)]



    def backprop(self,x,y):

        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation=x
        activations=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
        #backward pass
        delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())

        for l in range(2,self.num_layers):
            z=zs[-1]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta*sp)
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def evaluate(self,test_data):
        #return the number of test inpus for which the neural network outputs the
        #correct result.note that the neural network's output is assumed to be the index of
        #whichever neuron in the final layer has the highest activation.
        test_results=[(np.argmax(self.feedforward(x)),y)
                      for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def cost_derivative(self,output_activations,y):
         return (output_activations-y)






def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """derivative of sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))



'''
import mnist_loader
training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
import network
net=network.Network([784,30,10])
net.SGD(training_data,30,10,1.0,test_data=test_data)
'''




