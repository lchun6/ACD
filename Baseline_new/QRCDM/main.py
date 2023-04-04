#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: Bean
# datetime:2023/2/8 15:36
from QRCDM_model import QRCDM
from Test_Model import test_model
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append("../..")
from initial_dataSet2 import DataSet

if __name__ == '__main__':
    # ----------基本参数--------------
    basedir = '../../'
    dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC', 'FrcSub')
    # epochs_list = (3, 2, 1, 1, 2)
    epochs_list = (15, 2, 1, 1, 2)
    save_list = ('a0910/', 'a2017/', 'junyi/', 'math_ec/', 'frcsub/')

    dataSet_idx = 0
    test_ratio = 0.2
    learn_rate = 9e-3
    batch_size = 32


    data_set_name = dataSet_list[dataSet_idx]
    epochs = 15
    device = 'cuda'

    dataSet = DataSet(basedir, data_set_name)
    train_data, test_train, test_valid, train_stu_list, test_stu_list, test_record = dataSet.get_train_test(
        dataSet.record, test_ratio=test_ratio)
    Q = dataSet.get_exer_conc_adj()

    obj_prob_index = "All"

    train_loader = DataLoader(TensorDataset(torch.tensor(list(train_stu_list)).float()),
                              batch_size=batch_size, shuffle=True)

    model = QRCDM(Q=Q, lr=learn_rate, device=device)


