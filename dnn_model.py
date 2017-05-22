# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU


def create_model_lstm(max_features):
    """
        max_features = 正の整数．語彙数．入力データの最大インデックス + 1
        LSTMを使った学習。つまり時系列データを扱うもの
    """

    model = Sequential()
    """
    Embedding層は分散表現を行う層らしいです。
    """
    model.add(Embedding(max_features, output_dim=256))
    model.add(LSTM(128))

    #出力層
    """
    回帰問題を解いているので、もちろん出力層の次元数は1こ
    """
    model.add(Dense(1))
    """
    活性化関数はLinear
    https://cdn-ak.f.st-hatena.com/images/fotolife/i/imslotter/20170112/20170112005543.png
    """
    model.add(Activation('linear'))

    """
    目的関数は、mean_squared_error(mse)
    kerasで使える目的関数は以下
    https://keras.io/ja/objectives/

    mseとは平均二乗誤差
    http://www5e.biglobe.ne.jp/~emm386/2016/distance/euclid01.html

    rmspropは最適化アルゴリズムの一つ
    rmspropを選んだ理由は、RNNに向いているとかなんとか聞いたから。
    目的関数の値を最小にするための勾配の進み方

    http://postd.cc/optimizing-gradient-descent/
    """
    model.compile(loss='mse',
              optimizer='rmsprop')

    return model

def create_model_example2(input_shape):
    model = Sequential()
    """
    分散表現を行う層だった、Embeddingを抜いている
    """
    model.add(LSTM(128, input_shape=input_shape))

    #出力層
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mse',
              optimizer='rmsprop')

    return model

def create_model_example3(input_shape):
    model = Sequential()
    """
    自分でカスタムしてみる。
    """
    model.add(GRU(256, input_shape=input_shape))

    # ドロップアウト層
    model.add(Dropout(0.1))

    # 隠れ層に１個追加してみる
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #出力層
    model.add(Dense(1)) # 全結合NN
    model.add(Activation('linear'))

    model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return model
