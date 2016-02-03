# coding: UTF-8

import pickle
# reader = pickle.load(open('result/convolutional_network_best.pkl', 'rb'))
# => ImportError: Cuda not found. Cannot unpickle CudaNdarray

from pylearn2.utils import serial

import codecs
def ccc(name):
    if name.lower() == 'windows-31j':
        return codecs.lookup('utf-8')
codecs.register(ccc)

data = serial.load('../data/mnist/mnist.pkl', 'rb')
serial.save('../data/mnist/mnist_train_X.pkl', data[0][0])
serial.save('../data/mnist/mnist_train_y.pkl', data[0][1].reshape((-1, 1)))
serial.save('../data/mnist/mnist_valid_X.pkl', data[1][0])
serial.save('../data/mnist/mnist_valid_y.pkl', data[1][1].reshape((-1, 1)))
serial.save('../data/mnist/mnist_test_X.pkl', data[2][0])
serial.save('../data/mnist/mnist_test_y.pkl', data[2][1].reshape((-1, 1)))

# 上記のdata[0][0]などの実体は、ndarray
