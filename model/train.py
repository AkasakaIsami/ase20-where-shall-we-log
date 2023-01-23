import configparser
import os
from datetime import datetime
import time

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from torch.utils.data import DataLoader
from torchinfo import summary

from dataset import MyDataset
from model import Classifier
from util import float_to_percent, my_collate

import warnings

warnings.filterwarnings("ignore")


def train(train_dataset: MyDataset, dev_dataset: MyDataset):
    # 先读取训练配置
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    USE_GPU = cf.getboolean('environment', 'useGPU') and torch.cuda.is_available()

    EPOCH = cf.getint('train', 'epoch')
    BATCH_SIZE = cf.getint('train', 'batchSize')
    LR = cf.getfloat('train', 'learningRate')
    EMBEDDING_DIM = cf.getint('embedding', 'dim')
    HIDDEN_DIM = cf.getint('train', 'hidden_dim')
    DROP = cf.getfloat('train', 'dropout')

    # 要记录训练信息
    record_file_path = os.path.join(cf.get('data', 'dataDir'), cf.get('data', 'projectName'), 'record_file')
    if not os.path.exists(record_file_path):
        os.makedirs(record_file_path)

    start_time = datetime.now()
    start_time_str = datetime.strftime(start_time, '%Y-%m-%d_%H:%M:%S')

    # 定义日志保存文件
    record_file_name = start_time_str + '_train_info_' + '.txt'
    record_file = open(os.path.join(record_file_path, record_file_name), 'w')
    record_file.write(f"本次实验开始时间：{start_time_str}\n")
    record_file.write(f"数据集信息：\n")
    record_file.write(f"    -数据集切分比：{cf.get('data', 'ratio')}\n")

    record_file.write(f"    -训练集正负样本数据量比： 1:{cf.getint('sample', 'PosNegRatio')}\n")
    record_file.write(f"    -训练集长度：{len(train_dataset)}\n")
    record_file.write(f"    -验证集长度：{len(dev_dataset)}\n")

    record_file.write(f"模型配置如下：\n")
    record_file.write(f"    - EPOCHS：{EPOCH}\n")
    record_file.write(f"    - BATCH_SIZE：{BATCH_SIZE}\n")
    record_file.write(f"    - LEARNING_RATE：{LR}\n")
    record_file.write(f"    - 词嵌入维度：{EMBEDDING_DIM}\n")
    record_file.write(f"    - 隐藏层维度：{HIDDEN_DIM}\n")
    record_file.write(f"    - dropout率：{DROP}\n")

    # 正式开始训练！
    train_loader = DataLoader(dataset=train_dataset, collate_fn=my_collate, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, collate_fn=my_collate, batch_size=BATCH_SIZE, shuffle=True)

    model = Classifier(input_size=EMBEDDING_DIM,
                       hidden_size=HIDDEN_DIM,
                       output_size=1,
                       dropout=DROP)

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_function = torch.nn.MSELoss()

    if USE_GPU:
        model = model.cuda()

    # 用于寻找效果最好的模型
    best_acc = 0.0
    best_model = model

    record_file.write(f"模型结构如下：\n")
    record_file.write(str(summary(model)) + '\n')

    # 控制日志打印的一些参数
    total_train_step = 0
    start = time.time()
    record_file.write(f"开始训练！\n")
    for epoch in range(EPOCH):
        print(f'------------第 {epoch + 1} 轮训练开始------------')
        record_file.write(f'------------第 {epoch + 1} 轮训练开始------------\n')
        model.train()
        for i, (x, y) in enumerate(train_loader):
            model.zero_grad()
            y_hat = model(x)
            loss = loss_function(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 1 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss.item()}")

            if total_train_step % 10 == 0:
                record_file.write(f"训练次数: {total_train_step}, Loss: {loss.item()}\n")

        total_val_loss = 0.0
        y_hat_total = torch.randn(0)
        y_total = torch.randn(0)

        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(dev_loader):
                y_hat = model(x)
                loss = loss_function(y_hat, y)

                total_val_loss += loss.item()

                y_trans = y.reshape(y.shape[0], )
                y_hat_trans = []
                for i in range(y_hat.shape[0]):
                    y_hat_trans.append(1 if y_hat[0].item() > 0.5 else 0)
                y_hat_trans = torch.tensor(y_hat_trans)

                y_hat_total = torch.cat([y_hat_total, y_hat_trans])
                y_total = torch.cat([y_total, y_trans])

        print(f"验证集整体Loss: {total_val_loss}")
        record_file.write(f"验证集整体Loss: {total_val_loss}\n")

        acc = accuracy_score(y_total.cpu(), y_hat_total.cpu())
        balanced_acc = balanced_accuracy_score(y_total.cpu(), y_hat_total.cpu())
        ps = precision_score(y_total.cpu(), y_hat_total.cpu())
        rc = recall_score(y_total.cpu(), y_hat_total.cpu())
        f1 = f1_score(y_total.cpu(), y_hat_total.cpu())
        c = confusion_matrix(y_total.cpu(), y_hat_total.cpu(), labels=[0, 1])

        print(f"验证集 accuracy_score: {float_to_percent(acc)}")
        print(f"验证集 balanced_accuracy_score: {float_to_percent(balanced_acc)}")
        print(f"验证集 precision_score: {float_to_percent(ps)}")
        print(f"验证集 recall_score: {float_to_percent(rc)}")
        print(f"验证集 f1_score: {float_to_percent(f1)}")
        print(f"验证集 混淆矩阵:\n {c}")

        record_file.write(f"验证集 accuracy_score: {float_to_percent(acc)}\n")
        record_file.write(f"验证集 balanced_accuracy_score: {float_to_percent(balanced_acc)}\n")
        record_file.write(f"验证集 precision_score: {float_to_percent(ps)}\n")
        record_file.write(f"验证集 recall_score: {float_to_percent(rc)}\n")
        record_file.write(f"验证集 f1_score: {float_to_percent(f1)}\n")
        record_file.write(f"验证集 混淆矩阵:\n {c}\n")

        # 主要看balanced_accuracy_score
        if balanced_acc > best_acc:
            record_file.write(f"***当前模型的平衡准确率表现最好，被记为表现最好的模型***\n")
            best_model = model
            best_acc = balanced_acc

    end = time.time()
    print(f"训练完成，共耗时{end - start}秒。最佳balanced accuracy是{float_to_percent(best_acc)}。现在开始保存数据...")
    record_file.write(f"训练完成，共耗时{end - start}秒。最佳balanced accuracy是{best_acc}\n")
    record_file.write(
        f"——————————只有看到这条语句，并且对应的模型文件也成功保存了，这个日志文件的内容才有效！（不然就是中断了）——————————\n")
    record_file.close()

    model_file_name = start_time_str + '_model@' + float_to_percent(best_acc) + '.pth'
    model_save_path = os.path.join(record_file_path, model_file_name)
    torch.save(best_model, model_save_path)
    print('模型保存成功！')

    return best_model, os.path.join(record_file_path, record_file_name)
