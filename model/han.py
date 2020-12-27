import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
import string
import random
import re
import pandas as pd
from bs4 import BeautifulSoup
import os
from collections import defaultdict,Counter
import pickle

class WordAttention(nn.Module) : 
    
    def __init__(self,batch_size,hidden_size) : 
        
        super(WordAttention,self).__init__() 
        self.batch_size = batch_size
        self.linear = nn.Linear(hidden_size*2,hidden_size*2).to(device)
        self.word_proj_params = nn.Parameter(torch.Tensor(hidden_size*2,1)).to(device)
        self.initialize_weight()
        
    def initialize_weight(self) : 
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.word_proj_params)
        
    def forward(self,outputs) : 
        
        outputs = outputs.permute(1,0,2) #[batch_size, sent_len, hidden_dim*2]

        u = torch.tanh(self.linear(outputs)) #[batch_size, sent_len, hidden_dim*2]  
        word_proj_params = self.word_proj_params.expand(self.batch_size,-1,-1) #[batch_size,hidden_dim*2,1]
    
        atten = torch.bmm(u,word_proj_params) #[batch_size,sent_len,1]
        a = torch.softmax(atten,dim=1) #[batch_size,sent_len,1]
        s = torch.sum(torch.mul(a,outputs),dim=1) #[batch_size,hidden_dim*2]
        
        return s,a

class WordRNN(nn.Module) : 
    
    def __init__(self,batch_size,vocab_size,embed_size,hidden_size,num_layer,max_sent_len,weights_matrix) : 
        
        super(WordRNN,self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.gru_hidden_size = hidden_size
        self.num_layer = num_layer
        self.max_sent_len = max_sent_len
#         self.embeddings = nn.Embedding(vocab_size,embed_size,padding_idx = 1).to(device)
        self.embeddings = create_emb_layer(weights_matrix).to(device)
        # GloVe 로 Initialize만 시키고, Training이 가능하게 해줍니다.
        self.gru = nn.GRU(embed_size,hidden_size,num_layer,bidirectional=True).to(device)
        
        self.word_atten = WordAttention(batch_size,hidden_size).to(device)
        self.initialize_weight()
        
    def initialize_weight(self) : 
        for layer_p in gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.xavier_normal_(gru.__getattr__(p),)

    def forward(self,input_,hidden) : 
        
        sent_vec_ls = []; word_attention_ls = []
        
        for i in range(self.max_sent_len) : 
            x = input_[:,i,:]  # x : [batch_size, T :(word length per sentence)]
            embeds = self.embeddings(x).permute(1,0,2) # [T, batch_size, embed_dim] 

            outputs, hidden = self.gru(embeds,hidden)
            
            sent_vec,word_attention = self.word_atten(outputs)
        
            sent_vec_ls.append(sent_vec.unsqueeze(1))
            word_attention_ls.append(word_attention.permute(0,2,1))
        
        sent_vec = torch.cat(sent_vec_ls,dim=1)
        word_attention = torch.cat(word_attention_ls,dim=1)
                
        return sent_vec,word_attention,hidden

class SentAttention(nn.Module) : 
    
    def __init__(self,batch_size,hidden_size) : 
        
        super(SentAttention,self).__init__() 
        self.batch_size = batch_size
        self.linear = nn.Linear(hidden_size*2,hidden_size*2).to(device)
        self.sent_proj_params = nn.Parameter(torch.Tensor(hidden_size*2,1)).to(device)
        self.initialize_weight()
        
    def initialize_weight(self) : 
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.sent_proj_params)
        
    def forward(self,outputs) : 
        
        outputs = outputs.permute(1,0,2) #[batch_size, doc_len, hidden_dim*2]
        u = torch.tanh(self.linear(outputs)) #[batch_size, doc_len, hidden_dim*2]
        sent_proj_params = self.sent_proj_params.expand(self.batch_size,-1,-1) #[batch_size,hidden_dim*2,1]
        atten = torch.bmm(u,sent_proj_params) #[batch_size,doc_len,1]
        a = torch.softmax(atten,dim=1) #[batch_size,doc_len,1]
        v = torch.sum(a * outputs,dim=1) #[batch_size,hidden_dim*2]
        return v,a

class SentRNN(nn.Module) : 
    
    def __init__(self,batch_size,vocab_size,embed_size,hidden_size,num_layer) : 
        
        super(SentRNN,self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.gru_hidden_size = hidden_size
        self.num_layer = num_layer
        
        self.gru = nn.GRU(hidden_size*2,hidden_size,num_layer,bidirectional=True).to(device)
        
        self.sent_atten = SentAttention(batch_size,hidden_size)
        self.initialize_weight()
        
    def initialize_weight(self) : 
        for layer_p in gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.xavier_normal_(gru.__getattr__(p),)

    def forward(self,x,hidden) : 
        
        x = x.permute(1,0,2) #x : [doc_len,batch_size, hidden*2]

        outputs, hidden = self.gru(x,hidden)
    
        doc_vec,sent_attention = self.sent_atten(outputs)
        
        return doc_vec,sent_attention,hidden
class HAN(nn.Module) : 
    
    def __init__(self,batch_size,vocab_size,embed_size,hidden_size,num_layer,max_sent_len,num_class,weights_matrix) : 
        
        super(HAN,self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layer
        self.max_sent_len = max_sent_len
        self.num_class = num_class
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        self.word_encoder =\
        WordRNN(batch_size,vocab_size,embed_size,hidden_size,num_layer,max_sent_len,weights_matrix).to(self.device)
        
        self.sent_encoder =\
        SentRNN(batch_size,vocab_size,embed_size,hidden_size,num_layer).to(self.device)
        
        self.proj_layer = nn.Linear(hidden_size*2,num_class).to(self.device)
        self.initialize_weight()
        
    def initialize_weight(self) : 
        torch.nn.init.xavier_uniform_(self.proj_layer.weight)
        
    def init_hidden(self,batch_size):
        hidden = \
        Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size, device=self.device))
            
        return hidden
    
    def forward(self,input_) : 
        
        (batch_size,sent_len,doc_len) = input_.size()
        
        word_encoder_hidden = self.init_hidden(batch_size)
        sent_vec,word_attention,hidden = self.word_encoder(input_,word_encoder_hidden)
        sent_vec = nn.LayerNorm(self.hidden_size*2).to(device)(sent_vec)
        
        sent_encoder_hidden = self.init_hidden(batch_size)
        doc_vec,sent_attention,hidden = self.sent_encoder(sent_vec,sent_encoder_hidden)
        doc_vec = nn.LayerNorm(self.hidden_size*2).to(device)(doc_vec)
        
        logit = self.proj_layer(doc_vec)
        log_softmax = torch.log_softmax(logit,dim=1)
        
        return log_softmax, word_attention, sent_attention