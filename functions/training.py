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

# **********************************************
# Loss function
# **********************************************
class LossFunction(object):
    def __init__(self, loss_calc_method=None):
        self.loss_calc_method = loss_calc_method

    def __call__(self, data, target):
        if self.loss_calc_method is 'mse':
            return self._mse(data, target)
        elif self.loss_calc_method is 'cross_entropy':
            return self._cross_entropy(data, target)
        else:
            return self._mse(data, target)

    # mse loss
    def _mse(self, data, target):
        return F.mse_loss(data, target)

    # cross entropy loss
    def _cross_entropy(self, data, target):
        return F.cross_entropy(data, target)

# **********************************************
# Training by trainer
# **********************************************
def training(
        data_loader, model, max_epoch,
        loss_calc_method=None,
        out_dir='result'):

    # make save dir
    os.makedirs(out_dir, exist_ok=True)

    # gpu setting
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # set up optimizer
    opt = optim.Adam(model.parameters())

    # loss func.
    if loss_calc_method == 'mse':
        loss_function = nn.MSELoss()
    elif loss_calc_method == 'cross_entropy': 
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = LossFunction(loss_calc_method)

    # trainer
    trainer = create_supervised_trainer(
            model, opt, loss_function, device=device)

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
    trainer.run(data_loader, max_epochs=max_epoch)
    
    # log output
    file_path = os.path.join(out_dir, 'log')
    file_ = open(file_path, 'w')
    json.dump(log, file_, indent=4)

    # gpu to cpu
    if device is not 'cpu':
        model.to('cpu')

    return model

