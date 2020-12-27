# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 2:52 PM
# @Author  : David Yuan
# @FileName: models.py
# @github  : https://github.com/hittle2015
# @Description:
import pickle
import json
import mmap
import jieba
import random
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict,List, Optional
from gensim.models import KeyedVectors
from utils import WordEmbedding

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def convert_to_jsonl(input_data_dir: Path, to_convert_files: List, out_file: Path):
    """
    input_data_dir: Path directory where raw train, valid or test text file reside
    to_convert_files:  List of grouped text files in different categories for labelling information
    out_file:  JSONL file  with id, text, label  attributes

    """
    text_length=[]
    writer = out_file.open('w')
    
    idx = 0

    for txt in to_convert_files:
        txt_file = input_data_dir / '{}'.format(txt)
      
        with txt_file.open('r') as f:
            for line in tqdm(f):
                if txt_file.name.endswith('pos.txt'):
                    label, text = int(1), line
                elif txt_file.name.endswith('neg.txt'):
                    label, text = int(0), line
                writer.write(json.dumps({
                            'id':idx,
                            'text': text,
                            'labels':[label]}, ensure_ascii=False) +'\n')
                idx+=1
                text_length.append(len(jieba.lcut(text)))
    writer.close()
    print('sentence length  : min:{},max:{},avg:{}'.format(min(text_length), max(text_length),
                                                          sum(text_length) / len(text_length)))
    print('total sample:{}'.format(idx))

def split_data(input_file: Path):
    train_file_writer = Path('../htqe_data/htqe_train.jsonl').open('w')
    valid_file_writer = Path('../htqe_data/htqe_valid.jsonl').open('w')
    test_file_writer = Path('../htqe_data/htqe_test.jsonl').open('w')   # learner translations only

    with input_file.open('r') as f:
        for idx, line in enumerate(f):
            if idx < 4800:
                train_file_writer.write(line)
            if 4800 <= idx:
                valid_file_writer.write(line)
            # if 55000 <= idx:
            #     test_file_writer.write(line)
    train_file_writer.close()
    valid_file_writer.close()
    test_file_writer.close()


def generate_data(input_files: List, with_test_data=False):
    """
    generating train, development and test data
    train: htqe_data/htqe_train.jsonl
    vailid：htqe_data/htqe_valid.jsonl
    test: htqe_data/htqe_test.jsonl
    """

    all_data = []
    total_num = 0
    for input_file in input_files:
        total_num  += get_num_lines(input_file)
        with Path(input_file).open('r') as f:
            for line in f:
                all_data.append(line)

    
    train_writer = Path('./htqe_data/htqe_train.jsonl').open('w')
    valid_writer = Path('./htqe_data/htqe_valid.jsonl').open('w')
    test_writer = Path('./htqe_data/htqe_test.jsonl').open('w')
 
    if with_test_data:
        split_num = round(int(total_num*0.8))
        upper_split_num = round(int(total_num*0.9))
        train_dev = all_data[:upper_split_num]
        random.shuffle(train_dev)
        train_data=train_dev
        valid_data=train_dev[split_num:upper_split_num]
        testing_data=all_data[upper_split_num:]
        random.shuffle(testing_data)

       
    
    for item in train_data:
        train_writer.write(item)
    print("number of training samples: {} ".format(len(train_data)))
    for item in valid_data:
        valid_writer.write(item)
    print("number of validating samples: {} ".format(len(valid_data)))
    for item in testing_data:
        test_writer.write(item)
    print("number of testing samples: {} ".format(len(testing_data)))
    train_writer.close()
    #valid_writer.close()
    test_writer.close()
    
           # train data
       



# merging two pretrained word/char vectors for src and tgt languages (EN-ZH)
def combine_Embeddings(src_embedding, tgt_embedding, out_embedding)->(str, List[float]):
    """
    src_embedding: filename of pretrained source language word/char vectors
    tgt_embedding: filename of pretrained target language word/char vectors
    out_embedding: filename of the merged vectors
    output: str:List[float]

    """
    src_wv, tgt_wv, out_wv = Path(src_embedding), Path(tgt_embedding), Path(out_embedding)
    with out_wv.open('w+') as embeddings:
        with src_wv.open() as swv,tgt_wv.open() as twv:
            
            src_vocab_size, src_vector_size = (int(float(x)) for x in swv.readline().split())
            tgt_vocab_size, tgt_vector_size = (int(float(x)) for x in twv.readline().split())
            vocab_size, vector_size = (src_vocab_size + tgt_vocab_size),src_vector_size 
            embeddings.write(str(vocab_size)+' '+str(vector_size)+'\n')
            for src_line in tqdm(swv, total=get_num_lines(src_wv)):
                embeddings.write(src_line)
            for tgt_line in tqdm(twv, total=get_num_lines(tgt_wv)):
                embeddings.write(tgt_line)
    

def make_word_embedding(input_files: List, word_embedding: str):
    # loading word embedding
    wv = KeyedVectors.load_word2vec_format(word_embedding, binary=False, encoding='utf-8', unicode_errors='ignore')

    word_set = set()
    # word vectors
    for input_file in input_files:
        input_file = Path(input_file)
        with input_file.open('r') as f:
            for line in tqdm(f):
                json_line = json.loads(line)
                word_set = word_set.union(set(jieba.lcut(json_line['text'])))



    stoi = defaultdict(int)
    itos = defaultdict(str)
    vectors = []
    for idx, word in enumerate(word_set):
        if word in wv.vocab:
            stoi[word] = len(stoi)
            itos[len(itos)] = word
            vectors.append(wv.get_vector(word))
    word_embedding = WordEmbedding(stoi=stoi, itos=itos, vectors=vectors)

    # char vectors
    char_set = set()
    for input_file in input_files:
            input_file = Path(input_file)
            with input_file.open('r') as f:
                for line in tqdm(f):
                    json_line = json.loads(line)
                    char_set = char_set.union(set(list(json_line['text'])))

    stoi = defaultdict(int)
    itos = defaultdict(str)
    vectors = []
    for idx, char in enumerate(char_set):
        if char in wv.vocab:
            stoi[char] = len(stoi)
            itos[len(itos)] = char
            vectors.append(wv.get_vector(char))

    char_embedding = WordEmbedding(stoi=stoi, itos=itos, vectors=vectors)

    word_embedding_cache = Path('./word_embedding/.cache/htqe_word_embedding.pkl').open('wb')
    char_embedding_cache = Path('./word_embedding/.cache/htqe_char_embedding.pkl').open('wb')
    pickle.dump(word_embedding, word_embedding_cache)
    pickle.dump(char_embedding, char_embedding_cache)
    word_embedding_cache.close()
    char_embedding_cache.close()

