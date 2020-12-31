# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 2:52 PM
# @Author  : David Yuan
# @FileName: base_dataset.py
# @github  : https://github.com/hittle2015
# @Description:

from transformers import BertTokenizer, AutoTokenizer,AutoConfig
from pathlib import Path
# import tfrecord
import random
from dataclasses import dataclass
from typing import List, Optional, Union
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset
import torch

@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: Optional[str]
    text: str
    label: int


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    # token_type_ids: Optional[List[int]] = None
    label: int

    def __post_init__(self):
        self.sent_len = len(self.input_ids)

class BaseDataSet(Dataset):
    """
    for loading all data in memory
    """

    def __init__(self, transformer_model, overwrite_cache,force_download,cache_dir):
        # 分词器
        if transformer_model:
            self.transformer_config = AutoConfig.from_pretrained(transformer_model,force_download=force_download,cache_dir=cache_dir)
            # self.tokenizer = AutoTokenizer.from_pretrained(transformer_model,force_download=force_download,cache_dir=cache_dir)
            self.tokenizer = BertTokenizer.from_pretrained(transformer_model,config=self.transformer_config,force_download=force_download,cache_dir=cache_dir)
        if not self.feature_cache_file.exists() or overwrite_cache:
            self.features = self.save_features_to_cache()
        else:
            self.features = self.load_features_from_cache()

    def read_examples_from_file(self):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def save_features_to_cache(self):
        features = self.convert_examples_to_features()
        if self.shuffle:
            random.shuffle(features)
        print('saving feature to cache file : {}...'.format(self.feature_cache_file))
        torch.save(features, self.feature_cache_file)
        return features

    def load_features_from_cache(self):
        print('loading features from cache file : {}...'.format(self.feature_cache_file))
        return torch.load(self.feature_cache_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def make_k_fold_data(self, k):
        """
        :param k:
        :return:
        """
        kf = KFold(n_splits=k)  # splitting data into groups
        index_collecter = []
        for train_index, valid_index in kf.split(self.features):
            index_collecter.append((train_index, valid_index))
        return index_collecter
