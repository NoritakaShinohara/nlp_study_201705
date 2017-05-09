# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU

# https://hogehuga.com/post-1464/

def create_model_lstm(max_features):
    """
        max_features = 正の整数．語彙数．入力データの最大インデックス + 1
        LSTMを使った学習。つまり時系列データを扱うもの
    """

    model = Sequential()
    model.add(Embedding(max_features, output_dim=256))
    model.add(LSTM(128))

    #出力層
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return model

def create_model_example2(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))

    #出力層
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return model

def create_model_example3(input_shape):
    model = Sequential()
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
