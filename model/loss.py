# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 2:52 PM
# @Author  : David Yuan
# @FileName: loss.py
# @github  : https://github.com/hittle2015
# @Description:
import torch.nn.functional as F
import torch
from torch import nn


def ce_loss(output, target):

    return F.cross_entropy(output, target)


def binary_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target.float())


