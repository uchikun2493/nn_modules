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
from rnn import BasicRNN

# import my funciton
sys.path.append('../functions')
from convert_dtype import np_to_dataloder, np_to_tensor
from training import training, LossFunction

# **********************************************
# sample
# **********************************************

# sin波のデータセットを作成
def make_sin_dataset(data_len, inputs=1):

    theta = np.linspace(0, 2 * np.pi, data_len + 1)
    data = np.sin(theta)

    d = []
    t = []
    for i in range(len(data) - inputs):
        d.append(data[i:i+inputs])
        t.append(data[i+inputs:i+inputs+1])

    d = np.array(d, dtype=np.float32)
    t = np.array(t, dtype=np.float32)
    
    return d, t

# main
def main():

    # -------------------------------------
    # init setting
    # -------------------------------------
    args = sys.argv
    train_mode = True 
    if 2 <= len(args):
        if args[1] == '-1':
            train_mode = False
 
    save_dir = '../output/rnn'
    os.makedirs(save_dir, exist_ok=True)

    # モデルの保存パス
    model_path = os.path.join(save_dir, 'rnn')

    # -------------------------------------
    # define parameters
    # -------------------------------------
    epoch = 100
    batch_size = 1

    # gpu setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # -------------------------------------
    # define model
    # -------------------------------------
    inputs = 1
    hidden = 10
    outputs = 1
    model = BasicRNN(inputs, hidden, outputs)

    # -------------------------------------
    # make dataset
    # -------------------------------------
    train_data, train_teach = make_sin_dataset(50, inputs=1)
    test_data = train_data
    test_teach = train_teach

    plt.plot(train_data.reshape(50))
    plt.plot(train_teach.reshape(50))
    plt.show()

    # -------------------------------------
    # conversion data: numpy -> tensor -> DataLoader
    # -------------------------------------
    r, c = train_data.shape
    train_data = train_data.reshape(r, 1, c)
    train_teach = train_teach.reshape(r, 1, c)

    train_loader = np_to_dataloder(
            train_data, train_teach, batch_size,
            device=device, shuffle=True)

    # -------------------------------------
    # training 
    # -------------------------------------
    epoch = 100
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
    test_data = test_data.reshape(len(test_data), 1, inputs)
    test_teach = test_teach.reshape(len(test_teach), 1, inputs)

    test_data_, _ = np_to_tensor(test_data, test_teach)
    y, h = model(test_data_, h_req=True)

    print(y.data.numpy())
    print(h[0].data.numpy())

if __name__ == '__main__':
    main()

