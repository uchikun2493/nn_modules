# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# **********************************************
# rnn class
# **********************************************
class BasicRNNCell(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(BasicRNNCell, self).__init__()
        self.l1 = nn.LSTMCell(inputs, hidden)
        self.l2 = nn.Linear(hidden, outputs)
        self.hidden = hidden

    # Forward
    def __call__(self, x, h):
        h = self.l1(x, h)
        y = self.l2(h)
        return y, h

    def reset_hidden(self):
        return torch.zeros(1, self.hidden)

class BasicRNN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(BasicRNN, self).__init__()
        self.l1 = nn.LSTM(inputs, hidden, num_layers=1)
        self.l2 = nn.Linear(hidden, outputs)

        self.n_layer = 1
        self.hidden = hidden

    # Forward
    def __call__(self, x, h_req=False):

        n_sample = 1
        # 隠れ層の初期状態
        h0 = torch.zeros(self.n_layer, n_sample, self.hidden)
        # メモリセルの初期状態
        c0 = torch.zeros(self.n_layer, n_sample, self.hidden)

        # out, h = self.l1(x, (h0, c0))
        out, h = self.l1(x)
        y = self.l2(out[-1])

        if h_req is False:
            return y
        else:
            return y, h



