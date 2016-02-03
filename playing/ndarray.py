# coding: UTF-8
from pylearn2.utils.mnist_ubyte import read_mnist_images
import numpy as np

# 参考：http://rest-term.com/archives/2999/

data = read_mnist_images("../data/mnist/train-images-idx3-ubyte", dtype='float32')
# data は、　numpy.ndarray　オブジェクト！
print data.size # 47040000（バイト）= 60000枚 * 28 画素 * 28画素
print data[0].size # 784 = 28 * 28
print data[0][0].size # 28

def info(arr):
    print arr.flags
    print arr.ndim # 3 次元数
    print arr.size #  全体の要素数：47040000
    print arr.shape # 各次元の要素数：(60000,28,28)
    print arr.dtype # 配列要素のデータ型：float32　（float64じゃないのは、Pythonが32bit型だから？？）
    print arr.itemsize # 一要素のバイト数（int32,float32 => 4, int64,float64 => 8, bool => 1）

info(data)
assert data.shape == (60000,28,28)

dataBool = read_mnist_images("../data/mnist/train-images-idx3-ubyte", dtype='bool')
info(dataBool)
print dataBool[0][0][0] # => False

testArr = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12]
])
info(testArr) # int32型になるのはPythonが32bit型だから？？
assert testArr.dtype == "int32"

# 型変換
testArr64 = testArr.astype(np.int64)
info(testArr64) # => int64 になった！一要素のバイト数もちゃんと8になっている！
assert testArr64.dtype == "int64"

# 初期値0で生成
zeroArr = np.zeros([3,4])
print zeroArr
info(zeroArr) # float64型になる。。。不思議
assert zeroArr.dtype == "float64"

np.save('testArr', testArr) # バイナリでファイル保存 => testArr.npy
loadedArr = np.load('testArr.npy') # それを読む

# assert testArr == loadedArr # これエラーになる！

print np.amax(testArr)
print np.amin(testArr)

print np.amax(data)
print np.amin(data)
print np.histogram(data)






