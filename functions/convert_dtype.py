# my default import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

# conversion data: dataset -> DataLoader
def tensor_to_dataloder(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# conversion data: tensor -> TensorDataset
def tensor_to_dataset(tensor_data, tensor_label):
    return torch.utils.data.TensorDataset(tensor_data, tensor_label)

# conversion data: numpy -> tensor
def np_to_tensor(data, label, device='cpu'):
    tensor_data = torch.Tensor(data)
    tensor_label = torch.Tensor(label)
    tensor_data.to(device)
    tensor_label.to(device)
    return tensor_data, tensor_label

# conversion data: numpy -> tensor -> TensorDataset -> DataLoader
def np_to_dataloder(data, label, batch_size, device='cpu', shuffle=True):
    tensor_data, tensor_label = np_to_tensor(data, label, device=device)
    dataset = tensor_to_dataset(tensor_data, tensor_label)
    return tensor_to_dataloder(dataset, batch_size, shuffle=shuffle)
    
