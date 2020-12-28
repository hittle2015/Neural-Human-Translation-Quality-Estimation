# -*- coding: utf-8 -*-


import torch


def prepare_pack_padded_sequence( inputs_words, seq_lengths,descending=True):
    """
    for rnn model
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = torch.index_select(inputs_words,0,indices)
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices
