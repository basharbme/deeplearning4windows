[http://deeplearning.net/software/theano/library/config.html](http://deeplearning.net/software/theano/library/config.html)  

##theano環境変数の設定方法は3種類

- シェルで指定  
  （例）```THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True'  python <myscript>.py```      
- .theanorcファイルをユーザルートディレクトリに置く
- python内で、```os.environ['THEANO_FLAGS']```に値をセットする