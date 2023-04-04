import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from typing import Union, Tuple, Optional

def cross_entropy_loss(pred_, labels):  # 交叉熵损失函数
    pred = pred_.clamp(1e-6, 1 - 1e-6)
    pred_log = torch.log(pred)
    one_minus_log = torch.log(1 - pred)
    loss = -1 * (labels * pred_log + (1 - labels) * one_minus_log)
    loss_mean = loss.mean()
    return loss_mean

def format_test_data(record, test_record):  # train record , test record
    train = [[], [], []]  # 学生,习题，得分
    test = [[], [], []]  # 学生,习题，得分
    stu_list = set(record.index)

    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        test_item = test_record.loc[[stu], 'item_id'].values - 1
        test_score = test_record.loc[[stu], 'score'].values

        train[0].append(stu - 1)
        train[1].append(stu_item)
        train[2].append(stu_score)

        test[0].extend([count] * len(test_item))
        test[1].extend(test_item)
        test[2].extend(test_score)
        count += 1
    return train, test

def evaluate_obj(pred_, label):
    pred = np.array(pred_).round()

    acc = metrics.accuracy_score(label, pred)
    try:
        auc = metrics.roc_auc_score(label, pred)
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


def evaluate_sub(pred, label):
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return rmse, mae

def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae

# def format_data(record, test_record, n_splits=5):
#     train = [[], []]  # 习题，得分
#     # valid = [[], [], []]
#     test = [[], [], []]# 学生,习题，得分
#     stu_list = set(record.index)
#
#     KF = KFold(n_splits=n_splits, shuffle=True)  # 5折交叉验证
#     count = 0
#     for stu in stu_list:
#         stu_item = record.loc[[stu], 'item_id'].values - 1
#         stu_score = record.loc[[stu], 'score'].values
#         if len(stu_item) >= n_splits:
#             test_item = test_record.loc[[stu], 'item_id'].values - 1
#             test_score = test_record.loc[[stu], 'score'].values
#             for train_prob, valid_prob in KF.split(stu_item):
#                 train[0].append(stu_item[train_prob])
#                 train[1].append(stu_score[train_prob])
#
#                 valid[0].extend([count] * len(valid_prob))
#                 valid[1].extend(stu_item[valid_prob])
#                 valid[2].extend(stu_score[valid_prob])
#                 test[0].extend([count] * len(test_item))
#                 test[1].extend(test_item)
#                 test[2].extend(test_score)
#                 count += 1
#     valid_data = []
#     valid_data.append(torch.tensor(valid[0]).long())
#     valid_data.append(torch.tensor(valid[1]).long())
#     valid_data.append(torch.tensor(valid[2]).float())
#
#     test_data = []
#     test_data.append(torch.tensor(test[0]).long())
#     test_data.append(torch.tensor(test[1]).long())
#     test_data.append(torch.tensor(test[2]).float())
#
#     return train, valid_data, test_data
def format_data(record, n_splits=5):  # record x*2 x对应betch_size个学生的答题记录
    train = [[], [], []]  # 学生,习题，得分
    label = [[], [], []]  # 学生,习题，得分
    stu_list = set(record.index)  # 学生id list  大小为batch_size

    KF = KFold(n_splits=n_splits, shuffle=True)  # 2折交叉验证
    count = 0

    """
    一个batch可能包含多个学生，如果某个学生答题记录小于n_splits 就不划分 不过应该不存在这种情况
    KF.split(stu_item) 返回的是 索引 train_prob label_prob 维度是不固定的
    stu_item : 题号
    stu_score : 分数
    """
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1  # 考虑从0开始
        stu_score = record.loc[[stu], 'score'].values
        if len(stu_item) >= n_splits:

            for train_prob, label_prob in KF.split(stu_item):  # 对n_split批数据进行append
                train[0].append(stu - 1)
                train[1].append(stu_item[train_prob])
                train[2].append(stu_score[train_prob])

                label[0].extend([count] * len(label_prob))
                label[1].extend(stu_item[label_prob])
                label[2].extend(stu_score[label_prob])
                count += 1
    return train, label

def format_all_data(all_record):
    data = [[], [], []]  # 学生,习题，得分
    stu_list = set(all_record.index)

    for stu in stu_list:
        stu_item = all_record.loc[[stu], 'item_id'].values - 1
        stu_score = all_record.loc[[stu], 'score'].values

        data[0].append(stu - 1)
        data[1].append(stu_item)
        data[2].append(stu_score)

    return data

class QRCDM():
    def __init__(self, Q, student_num, concept_num, exercise_num, lr=1e-3, device='cpu'):
        self.device = torch.device(device)
        self.sigmoid = torch.nn.Sigmoid()
        self.skill_num = Q.shape[1]
        self.student_num = student_num
        self.concept_num = concept_num
        self.exercise_num = exercise_num
        # --------------模型参数---------------------
        Q = Q.to(device)
        W_ = Q.clone()
        W_.requires_grad = True
        D_ = Q.clone()
        D_.requires_grad = True
        # 猜测率、失误率
        guess_ = torch.ones((1, Q.shape[0])).to(device) * -2
        guess_.requires_grad = True
        miss_ = torch.ones((1, Q.shape[0])).to(device) * -2
        miss_.requires_grad = True
        # ------------------------------------------
        self.W_ = W_
        self.D_ = D_
        self.guess_ = guess_
        self.miss_ = miss_
        self.optimizer = torch.optim.Adam([self.W_, self.D_, self.guess_, self.miss_], lr=lr)

    def forward(self, score_list, prob_list):  # 前向传播,传入得分列表和习题索引列表
        k = self.skill_num
        device = self.device
        # drop = self.drop
        W_ = self.W_
        D_ = self.D_
        guess_ = self.guess_
        miss_ = self.miss_
        sigmoid = self.sigmoid

        A = torch.zeros(len(score_list), k).to(device)
        for i, X_i in enumerate(score_list):
            X_i = torch.tensor(X_i).float().to(device).reshape(1, -1)
            W_i = torch.softmax(W_[prob_list[i]], dim=0)
            A[i] = X_i @ W_i
        D = torch.softmax(D_, dim=1)
        Y_ = A @ D.T
        miss = sigmoid(miss_)
        guess = sigmoid(guess_)
        Y = (1 - miss) * Y_ + guess * (1 - Y_)
        return A, Y

    def train_model(self, train_index_loader: DataLoader,test_index_loader: DataLoader, train_df: pd.DataFrame, epochs: int = 5,
            n_splits: int = 5, test_train_df: pd.DataFrame = None ,test_valid_df: pd.DataFrame = None) -> None:
        device = self.device
        optimizer = self.optimizer

        for epoch in range(1, epochs + 1):
            test_loss_list = []
            # [[train_data],[valid_data]]
            obj_true_list, obj_pred_list = [[], []], [[], []]

            for betch_data in tqdm(train_index_loader, "[Epoch:%s]" % epoch):
                stu_list = np.array([x.numpy()
                                    for x in betch_data], dtype='int').reshape(-1)
                train_data, label_data = format_data(
                    train_df.loc[stu_list, :], n_splits=n_splits)

                # ----------训练集（起始）--------------
                _, pred = self.forward(train_data[2], train_data[1])
                pred = pred[label_data[0], label_data[1]]
                label = torch.FloatTensor(label_data[2]).to(self.device)
                loss = cross_entropy_loss(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if test_train_df is not None:
                self.test(test_index_loader, test_train_df, test_valid_df)

    def test(self, test_index_loader: DataLoader, test_train_df: pd.DataFrame, test_valid_df: pd.DataFrame) -> Tuple[float, float, float, float]:
        test_pred_list, test_label_list = [], []
        for betch_data in tqdm(test_index_loader, "[Testing:]"):
            stu_list = np.array([x.numpy() for x in betch_data], dtype='int').reshape(-1)
            train, test = format_test_data(test_train_df.loc[stu_list, :],
                                           test_valid_df.loc[stu_list, :])
            with torch.no_grad():
                _, all_pred = self.forward(train[2], train[1])
                test_pred = all_pred[test[0], test[1]].clone().to('cpu').detach()
                test_pred_list.extend(test_pred.tolist())
                test_label_list.extend(test[2])
        acc, auc, rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\ttest_result: \tacc:%.6f, auc:%.6f, rmse:%.6f, mae:%.6f" % (acc, auc, rmse, mae))
        return acc, auc, rmse, mae

    def get_A_and_Y(self, index_loader: DataLoader, all_record: pd.DataFrame, test_stu_list: pd.DataFrame):
        # 所有学生的A和Y
        A = torch.empty((self.student_num, self.concept_num))
        Y = torch.empty((self.student_num, self.exercise_num))
        for betch_data in tqdm(index_loader, "[get_A_and_Y:]"):
            stu_list = np.array([x.numpy() for x in betch_data], dtype='int').reshape(-1)  # test stu
            data = format_all_data(all_record.loc[stu_list, :])
            with torch.no_grad():
                cogn_state, all_pred = self.forward(data[2], data[1])
                A[data[0], :] = cogn_state.cpu().detach()
                Y[data[0], :] = all_pred.cpu().detach()
        # 学生id-1 得到A的索引
        # return A[ [stu - 1 for stu in test_stu_list]], Y[[stu - 1 for stu in test_stu_list]]
        return A, Y[[stu - 1 for stu in test_stu_list]]
    def save_parameter(self, save_dir):
        # 存储参数F
        np.savetxt(save_dir + 'W_.txt', self.W_.cpu().detach().numpy())
        np.savetxt(save_dir + 'D_.txt', self.D_.cpu().detach().numpy())
        np.savetxt(save_dir + 'miss_.txt', self.miss_.cpu().detach().numpy())
        np.savetxt(save_dir + 'guess_.txt', self.guess_.cpu().detach().numpy())
        print('模型参数已成功保存！')
