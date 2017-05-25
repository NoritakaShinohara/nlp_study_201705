# -*- coding: utf-8 -*-
"""
    公開されている学習済みモデルを利用する。
    http://aial.shiroyagi.co.jp/2017/02/japanese-word2vec-model-builder/
"""
from gensim.models.word2vec import Word2Vec

model_path = 'word2vec.gensim.model'
model = Word2Vec.load(model_path)

"""
    単語ベクトルの確認
"""
model.wv['行く']


"""
    入力されたワードと同じベクトル関係にあるもの
"""
out=model.most_similar(positive=[u'コンティネンタルサーカス'])
for x in out:
    print(x[0],x[1])

"""
    コサイン類似度
"""
model.wv.similarity('打ち合わせ', '挨拶')
model.wv.similarity('打ち合わせ', '営業')
