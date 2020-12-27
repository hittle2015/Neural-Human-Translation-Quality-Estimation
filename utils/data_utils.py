# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 2:52 PM
# @Author  : David Yuan
# @FileName: models.py
# @github  : https://github.com/hittle2015
# @Description:
from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from gensim.models import KeyedVectors

@dataclass
class WordEmbedding:
    stoi: Dict
    itos: Dict
    vectors: List[List[float]]



def add_pad_unk(stoi:Dict,itos:Dict,vectors:List[List[float]],wv:KeyedVectors):
    stoi['PAD'] = len(stoi)
    itos[len(itos)] = 'PAD'
    vectors.append(np.random.randn(wv.vector_size))
    stoi['UNK'] = len(stoi)
    itos[len(itos)] = 'UNK'
    vectors.append(np.random.randn(wv.vector_size))


