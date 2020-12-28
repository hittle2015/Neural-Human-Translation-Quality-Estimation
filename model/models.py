# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 2:52 PM
# @Author  : David Yuan
# @FileName: models.py
# @github  : https://github.com/hittle2015
# @Description:
from base import BaseModel
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from torch import nn
from torch.nn import functional as F
import pickle
import torch
import numpy as np
import os
from pathlib import Path
from utils.model_utils import prepare_pack_padded_sequence



class FastText(BaseModel):
    def __init__(self, class_num, word_embedding, train, dropout):
        super().__init__()
        word_embedding = pickle.load(Path(word_embedding).open('rb'))
        self.embedding_size = len(word_embedding.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.asarray(word_embedding.vectors)),
                                                      freeze=(not train))
        self.fc = nn.Linear(self.embedding_size, class_num)

    def forward(self, text, _, text_lengths):
        # text = [batch size,sent len]
        embedded = self.embedding(text).float()
        # embedded = [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        # pooled = [batch size, embedding_dim]
        return self.fc(pooled), pooled


class TextCNN(BaseModel):
    def __init__(self, n_filters, filter_sizes, dropout, word_embedding, train, class_num):
        # n_filters
        super().__init__()
        word_embedding = pickle.load(Path(word_embedding).open('rb'))
        self.embedding_size = len(word_embedding.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.asarray(word_embedding.vectors)),
                                                      freeze=(not train))
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, self.embedding_size)) for fs in
             filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, class_num)
        # self.fc_ = nn.Linear(1,2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1).float()
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, int(conv.shape[2])).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        # cat = self.fc(cat)
        # cat = cat.unsqueeze(dim=-1)
        return self.fc(cat), cat


class TextCNN1d(BaseModel):
    def __init__(self, n_filters, filter_sizes, dropout, word_embedding, train, class_num):
        super().__init__()
        word_embedding = pickle.load(Path(word_embedding).open('rb'))
        self.embedding_size = len(word_embedding.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.asarray(word_embedding.vectors)),
                                                      freeze=(not train))
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embedding_size, out_channels=n_filters, kernel_size=fs) for fs in filter_sizes])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, class_num)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1).float()
        # embedded = [batch size, emb dim, sent len]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, int(conv.shape[2])).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat), cat


class RnnModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, class_num, n_layers, bidirectional, dropout, word_embedding, train,
                 batch_first):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        word_embedding = pickle.load(Path(word_embedding).open('rb'))
        self.embedding_size = len(word_embedding.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.asarray(word_embedding.vectors)),
                                                      freeze=(not train))

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc = nn.Linear(hidden_dim * n_layers, class_num)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, _, text_lengths):
        # sort sentences in the descending order
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text = [batch size,sent len]
        embedded = self.dropout(self.embedding(text)).float()
        # embedded = [batch size, sent len, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths, batch_first=self.batch_first)
        self.rnn.flatten_parameters()
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            # output (seq_len, batch, num_directions * hidden_size)
            # hidden (num_layers * num_directions, batch, hidden_size)
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # turn back into the original order 
        output = output[desorted_indices]
        # output = [batch_size,seq_len,hidden_dim * num_directionns ]
        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.mean(output, dim=1)
        fc_input = self.dropout(output + hidden)
        out = self.fc(fc_input)

        return out, fc_input


class RCNNModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, class_num, n_layers, bidirectional, dropout, word_embedding, train,
                 batch_first):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_first = batch_first
        word_embedding = pickle.load(Path(word_embedding).open('rb'))
        self.embedding_size = len(word_embedding.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.asarray(word_embedding.vectors)),
                                                      freeze=(not train))

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.dropout = nn.Dropout(self.dropout)
        # Linear layer to get convolutional output to be passed to Pooling layer
        self.fc_cat = nn.Linear(hidden_dim * 2 + self.embedding_size, self.hidden_dim)
        # Fully connect layer
        self.fc = nn.Linear(self.hidden_dim, class_num)
        # Softmax non-linearity
        self.softmax = nn.Softmax()
        
        
      

    def forward(self, text, _, text_lengths):
        # sort sentences length in descending order
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
    
        # text = [batch size,sent len]
        embedded = self.dropout(self.embedding(text)).float()
        
        # embedded = [sent len, batch size, emb _dim]
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths.cpu(), batch_first=self.batch_first)
        #print(packed_embedded.data.shape)
        # packed_output
        # hidden [batch_size, seq_len, embed_size +2*hidden_size]
        self.rnn.flatten_parameters()
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        #print(packed_output.data.shape)

    

        # unpack sequence
        # output [sent len, batch_size * n_layers * bi_direction]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # turn back into the original order
        output = output[desorted_indices]

        #print(output.shape)
        # output = [batch_size,seq_len,hidden_dim * num_directionns ]
        batch_size, max_seq_len, hidden_dim = output.shape


        # concating the output and embedded layer
        input_features = torch.cat([output, embedded], dim=2)
        #input_features.shape = [batch_size, seq_len, embed_size + 2*hidden]
        #print(input_features.shape)
  

        # 
        output = torch.tanh(self.fc_cat(input_features))
        #linear_output.shape = (batch_size, seq_len, hidden_size)
        #print(output.shape)


        output = output.permute(0,2,1) # reshape for maxpooling
        #print(output.shape[2])
        output = F.max_pool1d(output, int(output.shape[2])).squeeze(2)
     
        #max out features[batch_size, hidden_dim]
        output = self.dropout(output)

        return self.fc(output), output


class RnnAttentionModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, class_num, n_layers, bidirectional, dropout, word_embedding, train,
                 batch_first):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        word_embedding = pickle.load(Path(word_embedding).open('rb'))
        self.embedding_size = len(word_embedding.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.asarray(word_embedding.vectors)),
                                                      freeze=(not train))

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(self.hidden_dim * 2,self.hidden_dim*2))
        self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)

        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
            self.fc = nn.Linear(hidden_dim * 2, class_num)
        else:
            self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
            self.fc = nn.Linear(hidden_dim, class_num)

    def forward(self, text, _, text_lengths):
        # reorder the input in the descending sentence lengths
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths)
        # text = [batch size,sent len]
        embedded = self.dropout(self.embedding(text)).to(torch.float32)
        # embedded = [batch size,sent len,  emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths.cpu(), batch_first=self.batch_first)

        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output = [sent len, batch size, hidden dim * num_direction]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # turning back to the original order
        output = output[desorted_indices]
        # attention
        # M = [sent len, batch size, hidden dim * num_direction]
        # M = self.tanh1(output)
        alpha = F.softmax(torch.matmul(self.tanh1(output), self.w), dim=0).unsqueeze(-1)  # dim=0表示针对文本中的每个词的输出softmax
        output_attention = output * alpha

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)

        output_attention = torch.sum(output_attention, dim=1)
        output = torch.sum(output, dim=1)

        fc_input = self.dropout(output * output_attention + hidden)
        # fc_input = self.dropout(output_attention)
        out = self.fc(fc_input)
        return out, fc_input


class DPCNN(nn.Module):
    def __init__(self, n_filters, class_num, word_embedding, train):
        super(DPCNN, self).__init__()

        word_embedding = pickle.load(Path(word_embedding).open('rb'))
        self.embedding_size = len(word_embedding.vectors[0])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.asarray(word_embedding.vectors)),
                                                      freeze=(not train))

        self.conv_region = nn.Conv2d(1, n_filters, (3, self.embedding_size), stride=1)
        self.conv = nn.Conv2d(n_filters, n_filters, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom

        self.relu = nn.ReLU()
        self.fc = nn.Linear(n_filters, class_num)

    def forward(self, text, _, text_lengths):
        # text [batch_size,seq_len]
        x = self.embedding(text)  # x=[batch_size,seq_len,embedding_dim]
        x = x.unsqueeze(1).to(torch.float32)  # [batch_size, 1, seq_len, embedding_dim]
        x = self.conv_region(x)  # x = [batch_size, num_filters, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)  # [batch_size, num_filters,1,1]
        x_embedding = x.squeeze()  # [batch_size, num_filters]
        x = self.fc(x_embedding)  # [batch_size, 1]
        return x, x_embedding

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


class TransformersModel(BaseModel):

    def __init__(self, transformer_model, cache_dir, force_download, is_train,  class_num):
        super(TransformersModel, self).__init__()
        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        # whether to continue training the transformers params
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train

        self.fc = nn.Linear(self.transformer_model.config.to_dict()['hidden_size'], class_num)

    def forward(self, input_ids, attention_masks, text_lengths):
        sentence = self.transformer_model(input_ids, attention_mask=attention_masks)
        # sentence = torch.sum(sentence,dim=1)   # cls 分类能力不佳
        out, pooler = sentence.last_hidden_state, sentence.pooler_output
        batch_size, text_length, hidden_dim = sentence.last_hidden_state.shape
        # average/sum  (mean is better than sum!!!!) the last hidden layer as input for next linear layer ()
        output = torch.mean(torch.reshape(out, [batch_size, -1, hidden_dim]), dim=1)
        output = self.fc(output)
        return output, pooler

class TransformersCNN(nn.Module):

    def __init__(self, transformer_model, cache_dir, force_download, n_filters, filter_sizes, dropout,
                 is_train, class_num):
        super(TransformersCNN, self).__init__()
        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        # whether to continue training the transformers params
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train
        # cnn
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n_filters, (k, self.transformer_model.config.to_dict()['hidden_size'])) for k in
             filter_sizes])
        self.dropout = nn.Dropout(dropout)

        self.fc_cnn = nn.Linear(n_filters * len(filter_sizes), class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, int(x.size(2))).squeeze(2)
        return x

    def forward(self, input_ids, attention_masks, text_lengths):
        # context    
        # mask  the padding to be the same size as text length，padding as '0'，for instance：[1, 1, 1, 1, 0, 0]
        encoder_out = self.transformer_model(input_ids, attention_mask=attention_masks)
        # feeding the last hidden state
        encoder_out = self.dropout(encoder_out.last_hidden_state) 
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out_embedding = self.dropout(out)
        out = self.fc_cnn(out_embedding)
        return out, out_embedding


class TransformersRNN(nn.Module):

    def __init__(self, transformer_model, cache_dir, force_download, rnn_type, hidden_dim, n_layers, bidirectional,
                 batch_first, dropout,is_train, class_num):
        super(TransformersRNN, self).__init__()

        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir, 
                                                              force_download=force_download,
                                                              output_hidden_states=True, 
                                                              output_attentions=True)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        # whether to continue training the transformers params
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.transformer_model.config.to_dict()['hidden_size'],
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.transformer_model.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.transformer_model.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(hidden_dim * 2, class_num)

    def forward(self, input_ids, attention_masks, text_lengths):

        # text = [batch size,sent len]
        # context = input 
        # mask  the padding to be the same size as text length，padding as '0'：[1, 1, 1, 1, 0, 0]
        sentence_out = self.transformer_model(input_ids, attention_mask=attention_masks)
        # reorder sentences in the descending order 
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(sentence_out.last_hidden_state, text_lengths)
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths.cpu(),
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # output = [ batch size,sent len, hidden_dim * bidirectional]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.sum(output, dim=1)
        fc_input = self.dropout(output + hidden)
        out = self.fc_rnn(fc_input)
        print(out, type(out), out.shape)

        return out,fc_input


class TransformersRCNN(BaseModel):
    def __init__(self, transformer_model, cache_dir, force_download,rnn_type, hidden_dim, n_layers, bidirectional,batch_first,
                 dropout, class_num,is_train):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        # whether to continue training the transformers params
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.transformer_model.config.to_dict()['hidden_size'],
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.transformer_model.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.transformer_model.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        # self.maxpool = nn.MaxPool1d()
        self.fc = nn.Linear(hidden_dim * n_layers, class_num)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, input_ids, attention_masks, text_lengths):
        # text = [batch size,sent len]
        # context = input    
        # mask  the padding to be the same size as text length，padding as '0'：[1, 1, 1, 1, 0, 0]
        sentence_out= self.transformer_model(input_ids, attention_mask=attention_masks)
        sentence_out, sentence_len, cls = sentence_out.last_hidden_state, sentence_out.last_hidden_state.shape[1], sentence_out.pooler_out
        cls = cls.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        sentence_out = sentence_out + cls
        # descending order
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(sentence_out, text_lengths)
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths,
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]
        batch_size, max_seq_len, hidden_dim = output.shape
        out = torch.transpose(output.relu(), 1, 2)

        out_embedding = F.max_pool1d(out, int(max_seq_len)).squeeze()
        out = self.fc(out_embedding)
        print(out, type(out), out.shape)

        return out,out_embedding
