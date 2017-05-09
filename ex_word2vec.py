# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec

def word2vec_model():
    """
    有志が作ったwikipediaを元に学習済みのmodelをロードする

    https://github.com/shiroyagicorp/japanese-word2vec-model-builder
    分散ベクトルの出力は５０次元となっている。
    """
    model_path = 'word2vec.gensim.model'
    model = Word2Vec.load(model_path)
    return model
