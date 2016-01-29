# coding: UTF-8

import os

os.environ["PYLEARN2_DATA_PATH"] = os.path.dirname(os.getcwd()) + "/data"
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

# 参考
# http://qiita.com/fetaro/items/448407a6964d307e8840

import codecs
def ccc(name):
    if name.lower() == 'windows-31j':
        return codecs.lookup('utf-8')
codecs.register(ccc)

# handle = os.open("C:\\Users\\jgpua_000\\ml\\pylearn2_test\\data\\mnist\\train-images-idx3-ubyte", os.O_RDONLY)

from pylearn2.scripts.train import train
# train(os.path.join(pylearn2.__path__[0],"scripts/autoencoder_example/dae.yaml"))
train("mnist.yaml")
