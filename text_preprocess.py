# -*- coding: utf-8 -*-

import zipfile, io
import pandas as pd
import glob
from ex_tokenize import tokenize
from dnn_model import create_model

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

"""
    テキストの前処理の過程を理解する
"""

# テストデータの読み込み
df = pd.read_csv("data/chiebukuro.csv", names=('score', 'review'))
tokenized_text_list = [tokenize(texts) for texts in df.review]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_text_list)
seq = tokenizer.texts_to_sequences(tokenized_text_list)

X = sequence.pad_sequences(seq, maxlen=400)
Y = df.score

model = create_model(5000)
model.fit(X, Y, epochs=1000, shuffle=True)

# model.summary()
