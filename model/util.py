import re
from random import random

import torch
from torch.nn.utils.rnn import pad_sequence


def my_collate(batch):
    x = []
    y = []
    ids = []
    for data in batch:
        x.append(data[0])
        y.append([data[1]])
        ids.append(data[2])
    x = pad_sequence(x, batch_first=True)
    y = torch.tensor(y).float()
    return x, y, ids


def float_to_percent(num: float) -> str:
    """
    浮点到百分比表示 保留两位小数
    :param num: 要转换的浮点数
    :return: 百分比表示
    """
    return "%.2f%%" % (num * 100)


def random_unit(p: float):
    """
    以p概率执行某段函数
    :param p:
    :return:
    """
    R = random()
    if R < p:
        return True
    else:
        return False
