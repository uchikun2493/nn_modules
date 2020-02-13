# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# **********************************************
# rnn class
# **********************************************
class BasicRNN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(BasicRNN, self).__init__()
        self.l1 = nn.LSTMCell(inputs, hidden)
        self.l2 = nn.Linear(hidden, outputs)

    # Forward
    def __call__(self, x, h):
        h = self.l1(x))
        y = self.l2(h)
        return y, h

    def reset_hidden(self):
        return torch.zeros(1, self.hidden)

