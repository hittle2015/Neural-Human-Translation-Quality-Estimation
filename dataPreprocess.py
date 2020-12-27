from data import htqe_preprocess
from utils import WordEmbedding
from pathlib import Path
import os
import sys
def main():
    # input_file = Path('weibo_senti_100k.csv')
    # out_file = Path('weibo_senti_100k.jsonl')
    # convert_to_jsonl(input_file, out_file)
    #generate_al_data(Path('weibo_senti_100k.jsonl'))
    # make_word_embedding(Path('weibo_senti_100k.jsonl'), '../word_embedding/sgns.weibo.word')
    input_data_dir = Path ('./htqe')
    train_dev_to_convert_files =['dev_neg.txt', 'dev_pos.txt', 'train_neg.txt', 'train_pos.txt', 'ht_pos.txt']
    test_to_convert_files =['htqe_test_neg.txt', 'htqe_test_pos.txt']
    train_dev_out_file = Path('./htqe_data/htqe_train_dev.jsonl')
    test_out_file = Path('./htqe_data/htqe_test.jsonl')
    htqe_preprocess.convert_to_jsonl(input_data_dir, train_dev_to_convert_files, train_dev_out_file)
    #htqe_preprocess.generate_data(train_dev_out_file, test_data=False) # 85 percent for train 15 for development 
    htqe_preprocess.convert_to_jsonl(input_data_dir, test_to_convert_files, test_out_file) # converting test data
    htqe_preprocess.generate_data([train_dev_out_file,test_out_file], with_test_data=True) # shuffing 
    htqe_preprocess.combine_Embeddings("/home/david/Workspace/corpus/word2vec/wiki.en.align.vec.1", "/home/david/Workspace/corpus/word2vec/wiki.zh.align.vec.1", './htqe/en_zh.vec')
    htqe_preprocess.make_word_embedding([train_dev_out_file, test_out_file],'./htqe/en_zh.vec')

if __name__ == '__main__':
    main()
