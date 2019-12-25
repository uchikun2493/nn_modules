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

# from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer
from ignite.contrib.handlers.tensorboard_logger import *

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

# **********************************************
# Loss function
# **********************************************
class LossFunction(object):
    def __init__(self):
        pass
    def __call__(self, data, target):
        return F.mse_loss(data, target)

# **********************************************
# Training autoencoder by trainer
# **********************************************
def training_mlp(
        data, label, unit_size, max_epoch, batchsize,
        out_dir='result'):

    # make save dir
    os.makedirs(out_dir, exist_ok=True)

    # each unit size
    save_dir = '../output/result_autoencoder'
    inputs = unit_size[0]
    hidden = unit_size[1]
    outputs = unit_size[2]

    # gpu setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # conversion data: numpy -> tensor -> DataLoader
    train_loader = np_to_dataloder(
            data, label, batch_size, device=device, shuffle=True)

    # define the autoencoder model
    model = BasicMLP(inputs, hidden, outputs)
    opt = optim.Adam(model.parameters())

    # loss function
    criterion = LossFunction()

    # trainer
    trainer = create_supervised_trainer(
            model, opt, criterion, device=device)

    # log variables init.
    log = []
    loss_iter = []

    # compute per Iterator
    @trainer.on(Events.ITERATION_COMPLETED)
    def add_loss(engine):
        loss_iter.append(engine.state.output)
    
    # print log (each epoch)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_report(engine):
        loss = sum(loss_iter) / len(loss_iter)
        log.append({'epoch':engine.state.epoch,
                    'loss':loss})
        if engine.state.epoch == 1:
            print('epoch\t\tloss')
        print(f'{engine.state.epoch}\t\t{loss:.10f}')
        loss_iter.clear()

    # start training
    trainer.run(train_loader, max_epochs=max_epoch)
    
    # log output
    file_path = os.path.join(out_dir, 'log')
    file_ = open(file_path, 'w')
    json.dump(log, file_, indent=4)

    # gpu -> cpu
    if device is not 'cpu':
        model.to('cpu')

    return model

# **********************************************
# sample code
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
    data = iris.data
    label = iris.target
    train_data = data[0:100, :]
    train_label = label[0:100]
    test_data = data[100:, :]
    test_label = label[100:]

    train_label = train_label.reshape([len(train_label), 1])

    # -------------------------------------
    # define parameters
    # -------------------------------------
    epoch = 100
    batchsize = 50
    unit_size = [4, 3, 3]

    # -------------------------------------
    # training 
    # -------------------------------------

    if train_mode is True:
        model = training_mlp(
                train_data, train_label, unit_size, epoch, batchsize,
                out_dir=save_dir)
        torch.save(model.state_dict(), model_path)
    else:
        inputs = unit_size[0]
        hidden = unit_size[1]
        outputs = unit_size[2]
        model = BasicMLP(inputs, hidden, outputs)
        param = torch.load(model_path)
        model.load_state_dict(param)


if __name__ == '__main__':
    main()

