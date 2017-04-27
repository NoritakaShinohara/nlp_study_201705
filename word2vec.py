# -*- coding: utf-8 -*-
"""
    公開されている学習済みモデルを利用する。
    http://aial.shiroyagi.co.jp/2017/02/japanese-word2vec-model-builder/
"""
from gensim.models.word2vec import Word2Vec

model_path = 'word2vec.gensim.model'
model = Word2Vec.load(model_path)
