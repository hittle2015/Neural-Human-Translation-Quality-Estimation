# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 2:52 PM
# @Author  : David Yuan
# @FileName: models.py
# @github  : https://github.com/hittle2015
# @Description:

from utils import WordEmbedding
import torch
import numpy as np
from model import makeModel, makeLoss, makeMetrics, makeOptimizer, makeLrSchedule
from utils import ConfigParser
import yaml
import random

# fix random seeds for reproducibility
SEED = 123
torch.cuda.empty_cache()
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def main(config):
    from data_process import makeDataLoader
    # 
    from trainer.htqe_trainer import Trainer

    logger = config.get_logger('train')
    train_dataloader, valid_dataloader, test_dataloader = makeDataLoader(config)

    model = makeModel(config)
    logger.info(model)

    criterion = makeLoss(config)
    metrics = makeMetrics(config)

    optimizer = makeOptimizer(config, model)
    lr_scheduler = makeLrSchedule(config, optimizer, train_dataloader.dataset)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=train_dataloader,
                      valid_data_loader=valid_dataloader,
                      test_data_loader=test_dataloader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


def run(config_fname):
    with open(config_fname, 'r', encoding='utf8') as f:
        config_params = yaml.load(f, Loader=yaml.Loader)
        config_params['config_file_name'] = config_fname

    config = ConfigParser.from_args(config_params)
    main(config)


if __name__ == '__main__':
    #run('configs/binary_classification/word_embedding_text_cnn.yml')
    #run('configs/binary_classification/word_embedding_text_cnn_1d.yml')
    #run('configs/binary_classification/word_embedding_fast_text.yml')
    #run('configs/binary_classification/word_embedding_rnn.yml')
    #run('configs/binary_classification/word_embedding_cnn_lstm_concatenated.yml') #cnn and rnn concatenated
    #run('configs/binary_classification/word_embedding_cnn_lstm_stacked.yml') # cnn and rnn stacked
    run('configs/binary_classification/word_embedding_cnn_lstm_stacked_with_attention.yml') # cnn and rnn stacked with attention
    #run('configs/binary_classification/word_embedding_rcnn.yml')
    #run('configs/binary_classification/word_embedding_rnn_attention.yml')
    #run('configs/binary_classification/word_embedding_dpcnn.yml')

    #run('configs/binary_classification/transformers.yml')
    #run('configs/binary_classification/transformers_cnn.yml')
    #run('configs/binary_classification/transformers_rnn.yml')
    #run('configs/binary_classification/transformers_rcnn.yml')
