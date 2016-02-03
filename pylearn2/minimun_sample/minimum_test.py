# coding: UTF-8

import os,codecs,platform
from pylearn2.config import yaml_parse
from pylearn2.scripts.train import train
import numpy as np
from pylearn2.utils import serial

os.environ["PYLEARN2_DATA_PATH"] = os.path.dirname(os.getcwd())

if platform.system() == "Windows":
    os.environ['THEANO_FLAGS'] = "floatX=float32,device=cpu"
else:
    os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu"

def ccc(name):
    if name.lower() == 'windows-31j':
        return codecs.lookup('utf-8')
codecs.register(ccc)

# prepare training data


# topo_view = np.zeros([5,28,28])
topo_view = np.random.randint(0,1,(3,28,28)) # [0, 1)の範囲で5 * 28 * 28の行列を作る
m, r, c = topo_view.shape
assert r == 28
assert c == 28
topo_view = topo_view.reshape(m, r, c, 1) # そうか、これがデザイン行列化ってことだ！！！

serial.save("input.pkl", topo_view)
serial.save("label.pkl", np.array([[0],[1],[2]]))

yaml = open("minimum.yaml", 'r').read()
hyper_params = {'train_stop': 5,
                'valid_stop': 50050,
                'test_stop': 5,
                'batch_size': 3, # サンプル数の倍数である必要があるらしい？（なんかエラーになった）
                'output_channels_h2': 4,
                'output_channels_h3': 4,
                'max_epochs': 5,
                'save_path': 'result'
                }
yaml = yaml % (hyper_params)

train = yaml_parse.load(yaml)
train.main_loop()

# train("minimum.yaml")





