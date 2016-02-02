# coding: UTF-8

import os

os.environ["PYLEARN2_DATA_PATH"] = os.path.dirname(os.getcwd()) + "/data"
os.environ['THEANO_FLAGS'] = "floatX=float32,device=cpu"

# 参考
# http://qiita.com/fetaro/items/448407a6964d307e8840

import codecs
def ccc(name):
    if name.lower() == 'windows-31j':
        return codecs.lookup('utf-8')
codecs.register(ccc)

# test
# save_path = "hereIsPath"
# print "%(save_path)s/hitokun" % locals()
# print '%(save_path)s/hitokun' % locals()


import convnet.tests.test_convnet as tc

tc.test_convolutional_network()



