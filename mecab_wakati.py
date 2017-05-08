# -*- coding: utf-8 -*-

import MeCab

def wakati(text):
    """
    文書を分かち書きする。
    """
    wakati = MeCab.Tagger('-O wakati')
    p = wakati.parse(text)
    return p
