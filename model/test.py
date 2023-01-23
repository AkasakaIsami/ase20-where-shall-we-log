import configparser
import os

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from torch.utils.data import DataLoader

from util import my_collate, float_to_percent

import warnings

warnings.filterwarnings("ignore")


def test(model, test_dataset, record_file_path: str):
    record_file = open(os.path.join(record_file_path), 'a')

    cf = configparser.ConfigParser()
    cf.read('config.ini')
    BATCH_SIZE = cf.getint('train', 'batchSize')
    USE_GPU = cf.getboolean('environment', 'useGPU') and torch.cuda.is_available()

    test_loader = DataLoader(dataset=test_dataset, collate_fn=my_collate, batch_size=BATCH_SIZE, shuffle=False)

    y_hat_total = torch.randn(0)
    y_total = torch.randn(0)

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            y_hat = model(x)

            y_trans = y.reshape(y.shape[0], )
            y_hat_trans = []
            for i in range(y_hat.shape[0]):
                y_hat_trans.append(1 if y_hat[0].item() > 0.5 else 0)
            y_hat_trans = torch.tensor(y_hat_trans)

            y_hat_total = torch.cat([y_hat_total, y_hat_trans])
            y_total = torch.cat([y_total, y_trans])

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
    record_file.write(f"测试集 混淆矩阵:\n {c}")

    record_file.close()
