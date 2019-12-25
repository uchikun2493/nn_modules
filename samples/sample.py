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
import torch.optim as optim

from ignite.engine import Events, create_supervised_trainer
from ignite.contrib.handlers.tensorboard_logger import *

# import my network models
sys.path.append('../networks')
from mlp import BasicMLP

# import my funciton
sys.path.append('../functions')
from convert_dtype import np_to_dataloder, np_to_tensor
from training import training, LossFunction

# **********************************************
# sample
# **********************************************
def main():

    # -------------------------------------
    # init setting
    # -------------------------------------
    args = sys.argv
    train_mode = True 
    if 2 <= len(args):
        if args[1] == '-1':
            train_mode = False
 
    save_dir = '../output/mlp'
    os.makedirs(save_dir, exist_ok=True)

    # モデルの保存パス
    model_path = os.path.join(save_dir, 'mlp')

    # -------------------------------------
    # load iris dataset
    # -------------------------------------
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris.data.astype(np.float32)
    label = iris.target.astype(np.int64)

    perm = np.random.permutation(data.shape[0])
    train_data = data[perm[0:100], :]
    train_label = label[perm[0:100]]
    test_data = data[perm[100:], :]
    test_label = label[perm[100:]]

    # -------------------------------------
    # define parameters
    # -------------------------------------
    epoch = 100
    batch_size = 10

    # gpu setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # -------------------------------------
    # define model
    # -------------------------------------
    model = BasicMLP(4, 3, 3)

    # -------------------------------------
    # conversion data: numpy -> tensor -> DataLoader
    # -------------------------------------
    train_loader = np_to_dataloder(
            train_data, train_label, batch_size,
            device=device, shuffle=True)

    # -------------------------------------
    # training 
    # -------------------------------------
    if train_mode is True:
        model = training(
                train_loader, model, epoch,
                loss_calc_method='cross_entropy',
                out_dir=save_dir)
        torch.save(model.state_dict(), model_path)
    else:
        param = torch.load(model_path)
        model.load_state_dict(param)

    # -------------------------------------
    # test
    # -------------------------------------
    test_data_, _ = np_to_tensor(test_data, test_label)
    predict = model(test_data_).data.numpy()
   
    print(np.argmax(predict, axis=1))
    print(test_label)

if __name__ == '__main__':
    main()

