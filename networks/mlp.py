# my default import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import json
import math

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# import my funciton
sys.path.append('../functions')
import convert_dtype
from convert_dtype import np_to_dataloder

# **********************************************
# mlp class
# **********************************************
class BasicMLP(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(BasicMLP, self).__init__()
        self.l1 = nn.Linear(inputs, hidden)
        self.l2 = nn.Linear(hidden, outputs)

    # Forward
    def __call__(self, x):
        h = F.relu(self.l1(x))
        y = self.l2(h)
        return y

