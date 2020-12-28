# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 2:52 PM
# @Author  : David Yuan
# @FileName: metric.py
# @github  : https://github.com/hittle2015
# @Description:

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr


def tensor_to_numpy(preds, labels):
    return preds.cpu().detach().numpy(),labels.cpu().detach().numpy() # 用于多标签类别分类

def accuracy(preds, labels):
    """
    averaged accuracy on all labels，irrespect of data imbalance
    """
    preds, labels = tensor_to_numpy(preds, labels)
    acc = accuracy_score(labels, preds, normalize=True)
    return acc



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return (correct.sum().cpu() / torch.FloatTensor([y.shape[0]])).item()
