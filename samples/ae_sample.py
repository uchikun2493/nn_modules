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
from autoencoder import Autoencoder

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
 
    save_dir = '../output/ae'
    os.makedirs(save_dir, exist_ok=True)

    # モデルの保存パス
    model_path = os.path.join(save_dir, 'ae')

    # -------------------------------------
    # load iris dataset
    # -------------------------------------
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris.data.astype(np.float32)
    teach = iris.target.astype(np.int64)

    # normalization
    data = data / np.max(data)

    perm = np.random.permutation(data.shape[0])
    train_data = data[perm[0:100], :]
    train_teach = train_data
    test_data = data[perm[100:], :]
    test_teach = test_data

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
    model = Autoencoder(4, 2)

    # -------------------------------------
    # conversion data: numpy -> tensor -> DataLoader
    # -------------------------------------
    train_loader = np_to_dataloder(
            train_data, train_teach, batch_size,
            device=device, shuffle=True)

    # -------------------------------------
    # training 
    # -------------------------------------
    if train_mode is True:
        model = training(
                train_loader, model, epoch,
                loss_calc_method='mse',
                out_dir=save_dir)
        torch.save(model.state_dict(), model_path)
    else:
        param = torch.load(model_path)
        model.load_state_dict(param)

    # -------------------------------------
    # test
    # -------------------------------------
    test_data_, _ = np_to_tensor(test_data, test_teach)
    predict = model(test_data_).data.numpy()
   
    # print(predict)
    # print(test_data_.data.numpy())

if __name__ == '__main__':
    main()

