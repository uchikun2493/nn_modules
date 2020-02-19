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

# dataset
from make_dataset import load_iris

# **********************************************
# sample
# **********************************************
def main():

    # -------------------------------------
    # init setting
    # -------------------------------------
    train_mode = True 
    for cmd in sys.argv:
        if cmd == '-1':
            train_mode = False
 
    # gpu setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    save_dir = '../output/mlp'
    os.makedirs(save_dir, exist_ok=True)

    # モデルの保存パス
    model_path = os.path.join(save_dir, 'mlp')

    # -------------------------------------
    # training parameters
    # -------------------------------------
    epoch = 100
    batch_size = 10

    # -------------------------------------
    # define model
    # -------------------------------------
    inputs = 4
    hidden = 3
    outputs = 3
    model = BasicMLP(inputs, hidden, outputs)

    # -------------------------------------
    # load iris dataset
    # -------------------------------------
    train_data, train_teach, test_data, test_teach = load_iris()

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
                loss_calc_method='cross_entropy',
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
   
    print(np.argmax(predict, axis=1))
    print(test_teach)

if __name__ == '__main__':
    main()

