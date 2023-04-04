import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append("../")
from initial_dataSet2 import DataSet
from shaoModel2 import CICDM


if __name__ == '__main__':
    result_table = pd.DataFrame(columns=['dataSet', 'counter', 'test_ratio', 'batch_size', 'enlarge_size',
                                         'h', 'n_splits', 'acc', 'auc', 'rmse', 'mae'])
    table_index = 0

    basedir = '../'
    # dataSet_list = ('FrcSub', 'ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC')
    dataSet_list = ('MathEC')
    # epochs_list = (60, 47, 33, 1,1)
    epochs_list = (1)
    counters = 5

    for dataSet_idx in range(1):
        # ----------基本参数--------------
        learn_rate = 3e-3
        weight_decay = None

        data_set_name = 'MathEC'
        epochs = 1
        device = 'cuda'
        # ----------基本参数--------------

        for test_ratio, n_splits, enlarge_size, h, batch_size in \
            [(0.2, 2, 3, 64, 128)]:

            dataSet = DataSet(basedir, data_set_name)
            train_data, test_train , test_valid ,train_stu_list,test_stu_list,test_record = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
            exer_conc_adj = dataSet.get_exer_conc_adj()
            # conc_conc_adj = dataSet.get_conc_conc_adj()
            item_concept = dataSet.get_item_concept_df()
            total_stu_list = dataSet.total_stu_list

            for counter in range(counters):
                print('第{}次实验,counter={},数据集:{}'.format(table_index + 1, counter, data_set_name))

                model = CICDM(student_num=dataSet.student_num,
                              concept_num=dataSet.concept_num,
                              exercise_num=dataSet.exercise_num,
                              exer_conc_adj=exer_conc_adj,
                              item_concept=item_concept,
                              enlarge_size=enlarge_size,
                              h=h,
                              lr=learn_rate,
                              device=device)
                train_index_loader = DataLoader(TensorDataset(torch.tensor(list(train_stu_list)).float()),
                                                batch_size=batch_size, shuffle=True)
                test_index_loader = DataLoader(TensorDataset(torch.tensor(list(test_stu_list)).float()),
                                               batch_size=batch_size, shuffle=True)
                # index_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                #                           batch_size=batch_size, shuffle=True)

                model.fit(train_index_loader,test_index_loader, train_data, epochs=epochs, n_splits=n_splits, test_train_df=test_train,test_valid_df=test_valid)
                acc, auc, rmse, mae = model.test(test_index_loader, test_train, test_valid)

                result_table.loc[table_index, 'dataSet'] = data_set_name
                result_table.loc[table_index, 'counter'] = counter
                result_table.loc[table_index, 'test_ratio'] = test_ratio
                result_table.loc[table_index, 'batch_size'] = batch_size
                result_table.loc[table_index, 'n_splits'] = n_splits
                result_table.loc[table_index, 'enlarge_size'] = enlarge_size
                result_table.loc[table_index, 'h'] = h
                result_table.loc[table_index, 'acc'] = acc
                result_table.loc[table_index, 'auc'] = auc
                result_table.loc[table_index, 'rmse'] = rmse
                result_table.loc[table_index, 'mae'] = mae

                result_table.to_csv('MathEC_sensitive12.csv')

                table_index += 1
