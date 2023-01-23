import configparser
import os

import pandas as pd
import torch
from gensim.models import Word2Vec

from dataset import MyDataset
from test import test
from train import train

import warnings

warnings.filterwarnings("ignore")


def dictionary_and_embedding(raw_dir, project, embedding_size):
    """
    :param project: 输入要计算embedding的项目
    :param embedding_size: 要训练的词嵌入大小
    """
    # 暂时用负样本吧 负样本也蛮多的了
    corpus_file_path = os.path.join(raw_dir, str(0), 'data.txt')
    model_file_name = project + "_w2v_" + str(embedding_size) + '.model'

    save_path = os.path.join(raw_dir, model_file_name)
    if os.path.exists(save_path):
        return

    from gensim.models import word2vec

    corpus = word2vec.LineSentence(corpus_file_path)
    w2v = word2vec.Word2Vec(corpus, vector_size=embedding_size, workers=16, sg=1, min_count=3)
    w2v.save(save_path)


def preprocess_data(raw_dir: str, process_dir: str, project: str, embedding_dim):
    '''
    逻辑是 先看process_dir里存没存
    没存再处理
    '''
    # 先导入embedding矩阵
    word2vec_path = os.path.join(raw_dir, project + '_w2v_' + str(embedding_dim) + '.model')
    word2vec = Word2Vec.load(word2vec_path).wv
    embeddings = torch.from_numpy(word2vec.vectors)
    embeddings = torch.cat([embeddings, torch.zeros(1, embedding_dim)], dim=0)

    def word2vector(word: str):
        max_token = word2vec.vectors.shape[0]
        index = [word2vec.key_to_index[word] if word in word2vec.key_to_index else max_token]
        return embeddings[index]

    if os.path.exists(process_dir):
        return

    neg_datalist = pd.DataFrame(columns=['id', 'data', 'label'])
    pos_datalist = pd.DataFrame(columns=['id', 'data', 'label'])

    with open(os.path.join(raw_dir, str(0), 'data.txt'), 'r') as file:
        for line in file:
            id = line.split(' ', 1)[0]
            words = line.split(' ')[1:]
            vectors = torch.randn(0, embedding_dim)
            for word in words:
                vector = word2vector(word)
                vectors = torch.cat([vectors, vector], dim=0)

            neg_datalist.loc[len(neg_datalist)] = [id, vectors, 0]

    with open(os.path.join(raw_dir, str(1), 'data.txt'), 'r') as file:
        for line in file:
            id = line.split(' ', 1)[0]
            words = line.split(' ')[1:]
            vectors = torch.randn(0, embedding_dim)
            for word in words:
                vector = word2vector(word)
                vectors = torch.cat([vectors, vector], dim=0)

            pos_datalist.loc[len(pos_datalist)] = [id, vectors, 1]

    os.makedirs(process_dir)
    neg_datalist = neg_datalist.sample(frac=1)
    pos_datalist = pos_datalist.sample(frac=1)
    neg_datalist.to_pickle(os.path.join(process_dir, 'negative.pkl'))
    pos_datalist.to_pickle(os.path.join(process_dir, 'positive.pkl'))


def make_dataset(process_dir: str, ratio: str, p_n_ratio: int, p_increase_rate: int):
    neg_datalist_file_path = os.path.join(process_dir, 'negative.pkl')
    pos_datalist_file_path = os.path.join(process_dir, 'positive.pkl')

    if not (os.path.exists(neg_datalist_file_path)) or (not os.path.exists(pos_datalist_file_path)):
        print(f'缺少文件{neg_datalist_file_path}或{pos_datalist_file_path}')
        return

    neg_datalist = pd.read_pickle(neg_datalist_file_path)
    pos_datalist = pd.read_pickle(pos_datalist_file_path)

    mode = cf.getint('sample', 'mode')
    if mode == 0:

        # 先增加正样本量
        pos_inc = pos_datalist.sample(frac=p_increase_rate)
        pos_datalist = pos_datalist.append(pos_inc)

        # 然后把正样本装进三个数据集
        ratios = [int(r) for r in ratio.split(':')]
        data_num = len(pos_datalist)

        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        train = pos_datalist.iloc[:train_split]
        dev = pos_datalist.iloc[train_split:val_split]
        test = pos_datalist.iloc[val_split:]

        # 然后切分负样本
        data_num = data_num * p_n_ratio
        neg_datalist = neg_datalist.sample(n=data_num if data_num < len(neg_datalist) else len(neg_datalist))
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        train = train.append(neg_datalist.iloc[:train_split])
        dev = dev.append(neg_datalist.iloc[train_split:val_split])
        test = test.append(neg_datalist.iloc[val_split:])

        train = train.sample(frac=1)
        dev = dev.sample(frac=1)
        test = test.sample(frac=1)

        train_dataset = MyDataset(train)
        dev_dataset = MyDataset(dev)
        test_dataset = MyDataset(test)

        return train_dataset, dev_dataset, test_dataset

    else:
        ratios = [int(r) for r in ratio.split(':')]
        data_num = len(pos_datalist)

        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        train = pos_datalist.iloc[:train_split]
        dev = pos_datalist.iloc[train_split:val_split]
        test = pos_datalist.iloc[val_split:]

        pos_inc = pos_datalist.sample(frac=p_increase_rate)
        train = train.append(pos_inc)
        num_pos = len(train)
        num_neg = num_pos * p_n_ratio

        data_num = len(neg_datalist)

        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        temp_train = neg_datalist.iloc[:train_split]
        temp_train_size = len(temp_train)
        temp_train = temp_train.sample(n=num_neg if num_neg < temp_train_size else temp_train_size)
        train = train.append(temp_train)
        dev = dev.append(neg_datalist.iloc[train_split:val_split])
        test = test.append(neg_datalist.iloc[val_split:])

        train = train.sample(frac=1)
        dev = dev.sample(frac=1)
        test = test.sample(frac=1)

        train_dataset = MyDataset(train)
        dev_dataset = MyDataset(dev)
        test_dataset = MyDataset(test)

        return train_dataset, dev_dataset, test_dataset


# 读取配置
cf = configparser.ConfigParser()
cf.read('config.ini')

project = cf.get('data', 'projectName')
ratio = cf.get('data', 'ratio')
p_n_ratio = cf.getint('sample', 'PosNegRatio')
p_increase_rate = cf.getfloat('sample', 'PosIncreaseRate')

raw_dir = os.path.join(cf.get('data', 'dataDir'), project, 'raw')
process_dir = os.path.join(cf.get('data', 'dataDir'), project, 'process')

embedding_dim = cf.getint('embedding', 'dim')

print(f'开始数据预处理（目标项目为{project}）...')
print('step1: 词嵌入训练...')
dictionary_and_embedding(raw_dir, project, embedding_dim)

print('step2: 处理原始数据...')
preprocess_data(raw_dir, process_dir, project, embedding_dim)

print('step3: 制作数据集...')
train_dataset, dev_dataset, test_dataset = make_dataset(process_dir, ratio, p_n_ratio, p_increase_rate)

print('step4: 开始训练...')
model, record_file_path = train(train_dataset, dev_dataset)

print('step5: 开始测试...')
test(model, test_dataset, record_file_path)
