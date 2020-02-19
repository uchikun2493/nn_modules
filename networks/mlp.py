# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# **********************************************
# mlp class
# **********************************************

# 3 layer network
class BasicMLP(nn.Module):
    def __init__(self, inputs, hidden, outputs, act_function=F.relu):
        super(BasicMLP, self).__init__()
        self.l1 = nn.Linear(inputs, hidden)
        self.l2 = nn.Linear(hidden, outputs)

        # activation function
        self.act_function = act_function

    # Forward
    def __call__(self, x):
        h = self.act_function(self.l1(x))
        y = self.l2(h)
        return y

