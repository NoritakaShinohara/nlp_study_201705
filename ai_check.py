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
    print("決定係数", r2_score(y_train, y_train_pred))



"""
    予測結果を表示
"""
def print_predict_result(estimator, x_input, y_correct):
    # ROC
    scores = estimator.predict_proba(x_input)
    # print('roc_auc_score: %.3f' % roc_auc_score(y_true=y_correct, y_score=scores))
    print(scores)

    y_pred = estimator.predict(x_input)
    flatten_y_pred = list(chain.from_iterable(y_pred))

    print(y_correct)
    print(flatten_y_pred)

    # flatten_y_pred = list(map(lambda x: 1 if x >= 0.5 else 0, flatten_y_pred))
    # # print(x_input)
    # # print(y_correct)
    #
    # # list(map(lambda y: 1, y_pred))
    # # 正解率
    # print('Accuracy: %.3f' % accuracy_score(y_true=y_correct, y_pred=flatten_y_pred))
    # # # F1
    # print('F1: %.3f' % f1_score(y_true=y_correct, y_pred=flatten_y_pred))
    # # 適合率
    # print('Precision: %.3f' % precision_score(y_true=y_correct, y_pred=flatten_y_pred))
    # # 再現率
    # print('Recall: %.3f' % recall_score(y_true=y_correct, y_pred=flatten_y_pred))
    # # 混合行列
    # print(confusion_matrix(y_correct, flatten_y_pred))
