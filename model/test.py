import configparser
import os
import random

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from util import my_collate, float_to_percent

import warnings

warnings.filterwarnings("ignore")


def test(model, test_dataset, record_file_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    record_file = open(os.path.join(record_file_path), 'a')

    cf = configparser.ConfigParser()
    cf.read('config.ini')
    BATCH_SIZE = cf.getint('train', 'batchSize')

    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=my_collate,
                             sampler=ImbalancedDatasetSampler(test_dataset),
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    y_hat_total = torch.randn(0)
    y_total = torch.randn(0)
    TP = []
    TN = []
    FP = []

    model.eval()
    with torch.no_grad():
        for i, (x, y, ids) in enumerate(test_loader):
            y_hat = model(x.to(device))

            y_trans = y.reshape(y.shape[0], )
            y_hat_trans = []
            for i in range(y_hat.shape[0]):
                y_hat_trans.append(1 if y_hat[0].item() > 0.5 else 0)
            y_hat_trans = torch.tensor(y_hat_trans)

            y_hat_total = torch.cat([y_hat_total, y_hat_trans])
            y_total = torch.cat([y_total, y_trans])

            for j in range(y_hat.shape[0]):
                statement_id = ids[j]
                fac = y[j].item()
                pre = y_hat[j].item()

                if fac == pre:
                    if fac == 1:
                        TP.append(statement_id)
                else:
                    if fac == 1:
                        TN.append(statement_id)
                    else:
                        FP.append(statement_id)

    acc = accuracy_score(y_total, y_hat_total)
    balanced_acc = balanced_accuracy_score(y_total, y_hat_total)
    ps = precision_score(y_total, y_hat_total)
    rc = recall_score(y_total, y_hat_total)
    f1 = f1_score(y_total, y_hat_total)
    c = confusion_matrix(y_total, y_hat_total, labels=[0, 1])

    print(f"测试集 accuracy_score: {float_to_percent(acc)}")
    print(f"测试集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
    print(f"测试集 precision_score: {float_to_percent(ps)}")
    print(f"测试集 recall_score: {float_to_percent(rc)}")
    print(f"测试集 f1_score: {float_to_percent(f1)}")
    print(f"测试集 混淆矩阵:\n {c}")

    record_file.write("下面是测试集结果：\n")
    record_file.write(f"测试集 accuracy_score: {float_to_percent(acc)}\n")
    record_file.write(f"测试集 balanced_accuracy_score: {float_to_percent(balanced_acc)}\n")
    record_file.write(f"测试集 precision_score: {float_to_percent(ps)}\n")
    record_file.write(f"测试集 recall_score: {float_to_percent(rc)}\n")
    record_file.write(f"测试集 f1_score: {float_to_percent(f1)}\n")
    record_file.write(f"测试集 混淆矩阵:\n {c}\n")

    # 对于TP（猜对了）、TN（没猜出来）、FP（猜错了） 分别取20条写进文件
    index = 0
    record_file.write("预测正确的的TN有：\n")
    TP = random.sample(TP, 20 if len(TP) > 20 else len(TP))
    for item in TP:
        record_file.write(f'    -{index}. {item}\n')
        index += 1

    index = 0
    record_file.write("实际是正样本，却被预测为负样本的TN有：\n")
    TN = random.sample(TN, 20 if len(TN) > 20 else len(TN))
    for item in TN:
        record_file.write(f'    -{index}. {item}\n')
        index += 1

    index = 0
    record_file.write("实际是负样本，被预测为正样本的FP有：\n")
    FP = random.sample(FP, 20 if len(FP) > 20 else len(FP))
    for item in FP:
        record_file.write(f'    -{index}. {item}\n')
        index += 1

    record_file.close()
