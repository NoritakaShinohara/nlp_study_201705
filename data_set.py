# -*- coding: utf-8 -*-

import zipfile, io
import pandas as pd
import glob
from mecab_wakati import wakati
from dnn_model import create_model_lstm

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

def yahoo_reviews():
    """
    分かち書き済みのレビューと評価に分ける
    """
    df = pd.read_csv("data/chiebukuro.csv", names=('score', 'review'))
    reviews = [wakati(texts) for texts in df.review]
    return reviews, df.score


def tokenize_reviews(reviews):
    """
    kerasにデフォで入っている、トークン化メソッドでトークン化 （文章を配列にする）
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)
    seq = tokenizer.texts_to_sequences(reviews)

    X = sequence.pad_sequences(seq, maxlen=400)
    return X
