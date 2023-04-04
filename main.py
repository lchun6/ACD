import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from initial_dataSet import DataSet
from pModel import CICDM #导入模型
def save_param(save_dir, name, param):
    np.savetxt(save_dir + name, param.cpu().detach().numpy(), fmt='%.6f', delimiter=',')


if __name__ == '__main__':

    # ----------基本参数--------------
    basedir = './'
    dataSet_list = ('FrcSub','ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC')
    # epochs_list = (60, 47, 33, 1,1)
    epochs_list = (100, 100, 100, 1,3)

    dataSet_idx = 5
    test_ratio = 0.2
    batch_size = 64
    learn_rate = 3e-3
    n_splits = 2 #分成几份 K重交叉验证
    enlarge_size = 5
    h = 128
    data_set_name = dataSet_list[dataSet_idx]
    epochs = epochs_list[dataSet_idx]
    device = 'cuda'
    # device = ''
    # ----------基本参数--------------

    dataSet = DataSet(basedir, data_set_name)

    train_data, test_train , test_valid ,train_stu_list,test_stu_list,test_record = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio) #获取测试集和训练集的答题记录 user_id itemid score
    exer_conc_adj = dataSet.get_exer_conc_adj() # 获得q矩阵
    # conc_conc_adj = dataSet.get_conc_conc_adj()
    item_concept = dataSet.get_item_concept_df()
    # total_stu_list = (dataSet.total_stu_list)[0:int(len(dataSet.total_stu_list)*0.08)] #答题记录中所有答题的学生id
    total_stu_list = (dataSet.total_stu_list) #答题记录中所有答题的学生id

    model = CICDM(student_num=dataSet.student_num,
                  concept_num=dataSet.concept_num,
                  exercise_num=dataSet.exercise_num,
                  exer_conc_adj=exer_conc_adj,
                  item_concept=item_concept,
                  enlarge_size=enlarge_size,
                  h=h,
                  # conc_conc_adj=conc_conc_adj,
                  lr=learn_rate,
                  device=device)
    #根据答题记录的学生id 进行batch处理
    train_index_loader = DataLoader(TensorDataset(torch.tensor(list(train_stu_list)).float()),
                              batch_size=batch_size, shuffle=True)
    test_index_loader = DataLoader(TensorDataset(torch.tensor(list(test_stu_list)).float()),
                                    batch_size=batch_size, shuffle=True)
    #根据生成的test index 在生成 daset.record


    model.fit(train_index_loader,test_index_loader, train_data, epochs=epochs, n_splits=n_splits, test_train_df=test_train,test_valid_df=test_valid)
    acc, auc, rmse, mae = model.test(test_index_loader, test_train, test_valid)
    cognitive_state, score_pred = model.get_A_and_Y(test_index_loader, dataSet.record,test_stu_list)

    # 存储参数
    # save_param_dir = dataSet.save_parameter_dir
    # # save_param(save_param_dir, 'H.csv', torch.softmax(model.cd_net.conc_conc_w, dim=0))
    # save_param(save_param_dir, 'lambda.csv', torch.sigmoid(model.cd_net.lambd))
    #
    save_result_dir = dataSet.save_result_dir


    save_param(save_result_dir, 'cognitive_state.csv', cognitive_state)

    np.savetxt(save_result_dir + "test_record.csv", test_record.reset_index(), fmt='%.6f',delimiter=',')
