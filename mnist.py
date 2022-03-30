#!/usr/bin/env conda run -n vision python

#TODO: implement second gradient for the first layer in mlp_2

import numpy as np
import matplotlib.pyplot as plt
from mnist import data
from data_prep import one_hot
from activations import sigmoid, relu, swish

X_train, Y_train, X_test, Y_test = data()

Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)

class mlp_2:
    def __init__(self, X, Y, iterations=1000):
        self.w = []
        self.b = []
        self.f = []
        self.d = []
        self.l = -0.057
        self.h1 = 16
        self.h2 = 16
        self.h3 = 10
        self.BS = 10
        self.layers = [[784, 10], [10,10], [20,10]] 
        self.concat = [0, 0, 1]
        self.activation = [relu, relu, relu]
        loss = []
        self.dynamic(X, Y)

    def dynamic(self, X , Y ):
        loss = []
        X = (X - np.mean(X))/ np.std(X)
        self.dynamic_init()
        print(np.mean(self.w[2]))
        for j in range(iterations): 
            for i in range(len(self.layers)):
                self.dynamic_forward(X, i)
            for i in range(len(self.layers)):
                 self.dynamic_backward(X, Y, i)
            loss.append(np.mean((self.f[len(self.f) - 1] - Y[:self.BS]) ** 2)) 
        print(np.argmax(self.f[len(self.f) - 1], axis=1))
        print(np.argmax(Y[:self.BS], axis=1))
        print(loss[len(loss) - 1] )
        plt.plot(loss)
        plt.show()

    def dynamic_init(self):
        for i in range(len(self.layers)):
            self.w.append(np.random.uniform(0, 1, size=(self.layers[i][0], self.layers[i][1])) * 0.01)
            self.b.append(np.random.random((1, self.layers[i][1]))* 0.01)
            self.f.append(np.empty(()))
            self.d.append(np.empty(()))
    def dynamic_forward(self, X, i):
        if i == 0:
            self.f[i] = self.activation[i]((np.dot(X[:self.BS], self.w[i]) + self.b[i]))
        else:
            if self.concat[i] == 1: 
                concat = np.concatenate((self.f[i - 1], (self.f[i - 2]) * 1), axis=1)
            else:
                concat = self.f[i - 1]
            self.f[i] = self.activation[i]((np.dot(concat, self.w[i]) + self.b[i]))
        
    def dynamic_backward(self, X, Y, i):
        if i == 0:
            if self.concat[len(self.concat) - 1] == 1:
                concat = np.concatenate((self.f[len(self.f) - 2], self.f[len(self.f) - 3]),axis=1)           
            else:
                concat = self.f[len(self.f) - 2]
            mse_d = 2*(self.f[len(self.f) - 1] - Y[:self.BS])
            self.d[i] = mse_d * self.activation[len(self.activation) - 1](self.f[len(self.f) - 1], True)/self.BS
            grad_w = self.d[i].T @ concat 
            grad_b = np.mean(self.d[i], axis=0, keepdims=True)
            self.w[len(self.w) - 1] = self.w[len(self.w) - 1] + self.l * grad_w.T
            self.b[len(self.b) - 1] = self.b[len(self.b) - 1] + self.l * grad_b
        else:
            self.d[i]  = self.d[i - 1]  @ self.w[len(self.w) - i][:10].T * self.activation[len(self.activation) - 1 - i](self.f[len(self.f) - i - 1], True) 
            if i + 1 == len(self.layers):
                grad_w = self.d[i].T @ X[:self.BS]
                d2_2 = self.w[2][10:] @ self.f[0].T
                grad_w_2 = d2_2 @ X[:self.BS]
                #grad_w = np.mean((grad_w_2, grad_w), axis=0)

                d_1_2 = self.d[2].T @ X[:self.BS]
                grad_w = np.mean((d_1_2 , grad_w))
            else:
                grad_w = self.d[i].T @ self.f[len(self.f) - i - 2]
            grad_b = np.mean(self.d[i], axis=0, keepdims=True)
            
            
            self.w[len(self.w) - i - 1  ] = self.w[len(self.w) - i  - 1] + self.l * grad_w.T
            self.b[len(self.b)  - i - 1] = self.b[len(self.b)  - i - 1] + self.l * grad_b

    def param_init(self, input_shape):
        self.w.append(np.random.uniform(0, 1, size=(input_shape, self.h1)) * 0.01)
        self.w.append(np.random.uniform(0, 1, size=(self.h1, self.h2))* 0.01)
        self.w.append(np.random.uniform(0, 1, size=(self.h2, self.h3))*0.01)

        self.b.append(np.random.random((1, self.h1))* 0.01)
        self.b.append(np.random.random((1, self.h2))* 0.01)
        self.b.append(np.random.random((1, self.h3)) * 0.01) 

    
class mlp: 
    def __init__(self, X, Y, iterations=1000):
        self.w = []
        self.b = []
        self.f = []
        self.d = []
        self.l = -0.00037
        self.h1 = 16
        self.h2 = 16
        self.h3 = 10
        self.BS = 10
        self.layers = [[784, 10], [10,10], [10,10]]
        self.activation = [relu, relu, relu]
        loss = []

        self.dynamic(X, Y)

    def classic(self, X, Y):
        loss = []
        self.f.append(np.empty(()))
        self.f.append(np.empty(()))
        self.f.append(np.empty(()))
        #X = X/ 255

        self.param_init(784)
        for i in range(iterations):
            self.forward(X)
            self.backward(X, Y)
            loss.append(np.mean((self.f[len(self.f) - 1] - Y[:self.BS]) ** 2)) 
        print(loss[len(loss) - 1])
        print(loss[0])
        print(np.argmax(self.f[2], axis=1))
        print(np.argmax(Y[:self.BS], axis=1))
        plt.plot(loss)
        plt.show()
        

    def dynamic(self, X , Y ):
        loss = []
        #X = (X - np.mean(X[:self.BS]))/ np.std(X[:self.BS])
        self.dynamic_init()
        for j in range(iterations): 
            for i in range(len(self.layers)):
                self.dynamic_forward(X, i)
            for i in range(len(self.layers)):
                 self.dynamic_backward(X, Y, i)
            loss.append(np.mean((self.f[len(self.f) - 1] - Y[:self.BS]) ** 2)) 
            if j == 0:
                print(loss[len(loss) - 1])
        print(np.argmax(self.f[len(self.f) - 1], axis=1))
        print(np.argmax(Y[:self.BS], axis=1))
        print(loss[len(loss) - 1] )
        plt.plot(loss)
        plt.show()
     
    def dynamic_init(self):
        for i in range(len(self.layers)):
            self.w.append(np.random.uniform(0, 1, size=(self.layers[i][0], self.layers[i][1])) * 0.01)
            self.b.append(np.random.random((1, self.layers[i][1]))* 0.01)
            self.f.append(np.empty(()))
            self.d.append(np.empty(()))


    def dynamic_forward(self, X, i):
        if i == 0:
            self.f[i] = self.activation[i]((np.dot(X[:self.BS], self.w[i]) + self.b[i]))
        else:
            self.f[i] = self.activation[i]((np.dot(self.f[i - 1], self.w[i]) + self.b[i]))

    def dynamic_backward(self, X, Y, i):
        if i == 0:
            mse_d = 2*(self.f[len(self.f) - 1] - Y[:self.BS])
            self.d[i] = mse_d * self.activation[len(self.activation) - 1](self.f[len(self.f) - 1], True)/self.BS
            print(self.d[i].shape)
            grad_w = self.d[i].T @ self.f[len(self.f) - 2]
            grad_b = np.mean(self.d[i], axis=0, keepdims=True)
            self.w[len(self.w) - 1] = self.w[len(self.w) - 1] + self.l * grad_w.T
            self.b[len(self.b) - 1] = self.b[len(self.b) - 1] + self.l * grad_b
        else:

            self.d[i]  = self.d[i - 1]  @ self.w[len(self.w) - i].T * self.activation[len(self.activation) - 1 - i](self.f[len(self.f) - i - 1], True) 

            if i + 1 == len(self.layers):
                grad_w = self.d[i].T @ X[:self.BS]
            else:
                grad_w = self.d[i].T @ self.f[len(self.f) - i - 2]
            grad_b = np.mean(self.d[i], axis=0, keepdims=True)
            
            
            self.w[len(self.w) - i - 1  ] = self.w[len(self.w) - i  - 1] + self.l * grad_w.T
            self.b[len(self.b)  - i - 1] = self.b[len(self.b)  - i - 1] + self.l * grad_b

    def param_init(self, input_shape):
        self.w.append(np.random.uniform(0, 1, size=(input_shape, self.h1)) * 0.01)
        self.w.append(np.random.uniform(0, 1, size=(self.h1, self.h2))* 0.01)
        self.w.append(np.random.uniform(0, 1, size=(self.h2, self.h3))*0.01)

        self.b.append(np.random.random((1, self.h1))* 0.01)
        self.b.append(np.random.random((1, self.h2))* 0.01)
        self.b.append(np.random.random((1, self.h3)) * 0.01) 

    def forward(self, X):
        self.f[0] = relu(np.dot(X[:self.BS], self.w[0]) + self.b[0])
        self.f[1] = relu(np.dot(self.f[0], self.w[1]) + self.b[1])
        self.f[2] = sigmoid(np.dot(self.f[1], self.w[2]) + self.b[2])

    def backward(self, X, Y):
        mse_d = 2*(self.f[len(self.f) - 1] - Y[:self.BS])

        d2 = mse_d * sigmoid(self.f[2], True)/self.BS
        d1 = d2 @ self.w[2].T * relu(self.f[1], True) 
        d0 = d1 @ self.w[1].T * relu(self.f[0], True)

        grad_w2 = d2.T @ self.f[1]
        grad_b2 = np.mean(d2, axis=0, keepdims=True)
        
        grad_w1 = d1.T @ self.f[0] 
        grad_b1 = np.mean(d1, axis=0, keepdims=True)
        
        grad_w0 = d0.T @ X[:self.BS]
        grad_b0 = np.mean(d0, axis=0, keepdims=True)


        self.w[2] = self.w[2] + self.l * grad_w2.T
        self.b[2] = self.b[2] + self.l * grad_b2

        self.w[1] = self.w[1] + self.l * grad_w1.T
        self.b[1] = self.b[1] + self.l * grad_b1 

        self.w[0] = self.w[0] + self.l * grad_w0.T
        self.b[0] = self.b[0] + self.l * grad_b0

   
        
        

iterations = 4000
mlp(X_train.reshape(-1, 784), Y_train, iterations=iterations)
#mlp_2(X_train.reshape(-1, 784), Y_train, iterations=iterations)


