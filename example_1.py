# -*- coding: utf-8 -*-

import zipfile, io
import pandas as pd
import glob
from dnn_model import create_model_lstm
from ai_check import print_predict_result, evaluate_regression
from data_set import yahoo_reviews, tokenize_reviews

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

"""
X: レビューの文章
Y: スコア

にデータを分けた後、Kerasにもともと入っているトークン化のメソッドを使って
学習させていきます。
"""

X, Y = yahoo_reviews()
X_tokenize = tokenize_reviews(X)

# 5,000という数字は適当です。（入力語彙数）
model = create_model_lstm(5000)
"""
epoch数は、一つの訓練データセットを何回繰り返して学習させるか
validation_splitは、学習データ、検証用データにどの割合で分けるか
"""
model.fit(X_tokenize, Y, epochs=15, shuffle=True, validation_split=0.1)

y_pred = model.predict(X_tokenize)

# AIと実際のデータにどれくらいの差があるか
evaluate_regression(Y, y_pred)
