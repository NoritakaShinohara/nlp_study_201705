# -*- coding: utf-8 -*-

import os
import sys

from keras.utils import np_utils

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing

import numpy as np  # 数値計算ライブラリ
import pandas as pd  # dataFrameを扱うライブラリ

# 結果算出
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import mean_squared_error, r2_score

from itertools import chain

def evaluate_regression(y_train, y_train_pred):
    """
    回帰モデルの評価を行う。
    評価関数は平均二乗誤差、決定係数
    """
    print("平均二乗誤差", mean_squared_error(y_train, y_train_pred))

    """
    決定係数
    1に近いほど良い値
    """
    print("決定係数", r2_score(y_train, y_train_pred))
