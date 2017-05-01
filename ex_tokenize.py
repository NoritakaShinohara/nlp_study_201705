# -*- coding: utf-8 -*-

import MeCab

# テキストを分かち書きして返す


def tokenize(text):
    wakati = MeCab.Tagger('-O wakati')
    p = wakati.parse(text)
    return p
