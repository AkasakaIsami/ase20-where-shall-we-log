import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(Classifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.dropout = dropout

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)

        output, _ = self.lstm(x, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred
