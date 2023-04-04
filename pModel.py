# 学习者作答习题时的得分不仅仅受到知识点的影响，还受到学习者的其他技能的影响
# 组织形式：Y=(1-λ)*Y_A + λ*Y_B

# A：学习者对知识点的熟练程度，大小=N*K
# B：学习者除知识点外的其他技能，大小=N*8
# C：学习者在知识簇上的属性，大小=N*K
# H：知识点的交互，大小=K*K
# W：习题与知识点的权重矩阵，大小=J*K，其中元素sigmoid函数后再除以行/列累加和归一化
# D：习题与其他技能的权重，大小=J*8，其中元素用行/列softmax函数归一化
# lambda：其他技能对答题记录的影响权重
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold
from torch import nn, Tensor
from typing import Union, Tuple, Optional
from torch.utils.data import TensorDataset, DataLoader
from initial_dataSet2 import DataSet
torch.autograd.set_detect_anomaly(True)


def format_data(record, n_splits=5):  # record x*2 x对应betch_size个学生的答题记录
    train = [[], [], []]  # 学生,习题,得分
    label = [[], [], []]  # 学生,习题,得分
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


def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


class CICDM_Net(nn.Module):
    def __init__(self, concept_num: int, exercise_num: int, enlarge_size:int, h:int,potential_num: int,  exer_conc_adj: Tensor,item_concept,device: str = 'cpu') -> None:
        super().__init__()
        assert exer_conc_adj.size(0) == exercise_num and exer_conc_adj.size(1) == concept_num, 'exercise_concept adjacency matrix size wrong!'
        self.device = device

        self.h = h
        self.concept_num = concept_num
        self.exercise_num = exercise_num
        self.exer_conc_adj = exer_conc_adj
        self.enlarge_size = enlarge_size
        self.potential_num = potential_num

        self.exer_matrix = nn.Parameter(torch.randn((exercise_num, h)))  # nn.Parameter(torch.randn_like(exercise_num,h)) #习题初始矩阵
        self.conc_matrix = nn.Parameter(torch.randn((concept_num, h))) # nn.Parameter(torch.randn_like(exercise_num,h)) #习题初始矩阵
        self.pote_matrix = nn.Parameter(torch.randn((potential_num, h)))


        self.lambd = nn.Parameter(torch.ones((1, exercise_num)) * -2)
        #
        self.guess = nn.Parameter(torch.ones((1, exercise_num)) * -2)
        self.slide = nn.Parameter(torch.ones((1, exercise_num)) * -2)
        self.attn_mask = self.exer_conc_adj.data.eq(0)

    def cos_sim(self,mat1, mat2, enlarge_size):
        up = mat1 @ mat2.T
        down = (((mat1 ** 2).sum(dim=1) ** (1 / 2)).reshape(-1, 1)) @ (
            ((mat2 ** 2).sum(dim=1) ** (1 / 2)).reshape(-1, 1)).T
        return (up / down)*enlarge_size


    def mnm(self):
        # self.exertoconc[self.exer_idx_list,self.conc_idx_list] = (self.exer_matrix[self.exer_idx_list]*self.conc_matrix[self.conc_idx_list]).sum(dim=1)
        self.exertoconc = self.cos_sim(self.exer_matrix,self.conc_matrix, self.enlarge_size).masked_fill(self.attn_mask, -1e9)  #
        self.conctoconc = self.cos_sim(self.conc_matrix,self.conc_matrix, self.enlarge_size)
        self.exertopote = self.cos_sim(self.exer_matrix,self.pote_matrix, self.enlarge_size)
    def forward(self, exer_list, score_list) -> Tuple[Tensor, Tensor]:  # exer_list、score_list的维度 batch_size *k重  * 答题记录
        self.mnm()
        A = torch.empty(len(exer_list), self.concept_num).to(self.device)  # congnitive  diagnosis
        # 获取知识点和习题的关系矩阵字典
        slide = torch.sigmoid(self.slide)
        guess = torch.sigmoid(self.guess)
        #-----p
        B = torch.empty(len(score_list), self.potential_num).to(
            self.device)  # batch_size * potential_num 潜在的习题对应的知识点即 学生对潜在知识点的认知状态
        lambd = torch.sigmoid(self.lambd)  # 1*N
        #------

        for i, X_i in enumerate(score_list):  # 处理每个学生
            X_i = torch.tensor(X_i).float().to(self.device).reshape(1, -1)  # 索引i  X_i:score_list 1 0 value
            # --------Knowledge concept start---------------
            W1_i_ = self.exertoconc[exer_list[i]]  #
            exer_conc_adj_i = self.exer_conc_adj[exer_list[i]]
            exer_conc_adj_i_sum = exer_conc_adj_i.sum(dim=0)  # 对列求和 # The cumulative sum of concepts not involved is 0
            W1_i = W1_i_[:, exer_conc_adj_i_sum != 0]
            val = X_i @ torch.softmax(W1_i, dim=0)

            W2_i = self.conctoconc[exer_conc_adj_i_sum != 0]
            W2_i = torch.softmax(W2_i, dim=0)
            A[i] = val @ W2_i


            #-------p
            D1_i_ = self.exertopote[exer_list[i]]
            D1_i = torch.softmax(D1_i_, dim=0)
            B[i] = X_i @ D1_i
            #------

        Y_A = A @ torch.softmax(self.exertoconc.T,dim=0)

        Y_B = B @ torch.softmax(self.exertopote.T, dim=0)
        Y_ = (1 - lambd) * Y_A + lambd * Y_B

        Y_ = Y_.clamp(1e-8, 1 - 1e-8)  # 收缩
        Y = (1 - slide) * Y_ + guess * (1 - Y_)
        return A, Y


class CICDM():
    def __init__(self, student_num: int, concept_num: int, exercise_num: int, exer_conc_adj: Tensor, item_concept :pd.DataFrame,
                 enlarge_size,potential_num,h,lr: float = 0.001, device: str = 'cpu') -> None:
        self.cd_net = CICDM_Net(concept_num, exercise_num, enlarge_size,h,potential_num,exer_conc_adj.to(device),item_concept, device=device).to(device)
        self.device = device
        self.cd_net.to(device)
        print(device)
        self.student_num = student_num
        self.concept_num = concept_num
        self.exercise_num = exercise_num
        self.potential_num = potential_num
        self.optimizer = torch.optim.Adam(self.cd_net.parameters(), lr=lr)
        self.loss = torch.nn.BCELoss(reduction='mean')

    def fit(self, train_index_loader: DataLoader,test_index_loader: DataLoader, train_df: pd.DataFrame, epochs: int = 5,
            n_splits: int = 5, test_train_df: pd.DataFrame = None ,test_valid_df: pd.DataFrame = None) -> None:
        # index_loader : userid loader 每批为batch_size
        # train_df 整个训练集
        # train_df.loc[stu_list, :] 从record中获取包含本batch学生的答题记录 大小不定
        # index_loader: 每次取batch size个学生的user_id迭代器
        acc, auc, rmse, mae = 0, 0, 0, 0
        for epoch in range(epochs):
            epoch_loss = 0
            # self.cd_net.mnm()

            for betch_data in tqdm(train_index_loader, "[Epoch:%s]" % (epoch + 1)):  # 每次返回一批user id 大小为batch size
                stu_list = np.array([x.numpy() for x in betch_data], dtype='int').reshape(-1)  # 把betch_data转为list 内容是userid
                train_data, label_data = format_data(train_df.loc[stu_list, :], n_splits=n_splits)  # 通过list userid 获取到对应users的答题记录 交叉验证 再次划分
                # 把一个batch的数据划分完 但是label_data和train_data的数据结构稍有不同
                # -----start training-------------------
                _, all_pred = self.cd_net(train_data[1], train_data[2])  # 题号、分数
                pred = all_pred[label_data[0], label_data[1]]
                label = torch.FloatTensor(label_data[2]).to(self.device)
                loss: Tensor = self.loss(pred, label)

                # ------start update parameters----------
                self.optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                loss.backward()
                self.optimizer.step()
                # ------ end update parameters-----------

            if test_train_df is not None:
                p_acc, p_auc, p_rmse, p_mae = self.test(test_index_loader, test_train_df, test_valid_df)
            if p_auc < auc:
                break
            else:
                auc = p_auc
    def test(self, test_index_loader: DataLoader, test_train_df: pd.DataFrame, test_valid_df: pd.DataFrame) -> Tuple[float, float, float, float]:
        test_pred_list, test_label_list = [], []
        for betch_data in tqdm(test_index_loader, "[Testing:]"):
            stu_list = np.array([x.numpy() for x in betch_data], dtype='int').reshape(-1)
            train, test = format_test_data(test_train_df.loc[stu_list, :],
                                           test_valid_df.loc[stu_list, :])
            with torch.no_grad():
                _, all_pred = self.cd_net(train[1], train[2])
                test_pred = all_pred[test[0], test[1]].clone().to('cpu').detach()
                test_pred_list.extend(test_pred.tolist())
                test_label_list.extend(test[2])
        acc, auc, rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\ttest_result: \tacc:%.6f, auc:%.6f, rmse:%.6f, mae:%.6f" % (acc, auc, rmse, mae))
        return acc, auc, rmse, mae

    def get_A_and_Y(self, index_loader: DataLoader, all_record: pd.DataFrame,test_stu_list: pd.DataFrame):
        # 所有学生的A和Y
        A = torch.empty((self.student_num, self.concept_num))
        Y = torch.empty((self.student_num, self.exercise_num))
        for betch_data in tqdm(index_loader, "[get_A_and_Y:]"):
            stu_list = np.array([x.numpy() for x in betch_data], dtype='int').reshape(-1) #test stu
            data = format_all_data(all_record.loc[stu_list, :])
            with torch.no_grad():
                cogn_state, all_pred = self.cd_net(data[1], data[2])
                A[data[0], :] = cogn_state.cpu().detach()
                Y[data[0], :] = all_pred.cpu().detach()
        #学生id-1 得到A的索引
        # return A[ [stu - 1 for stu in test_stu_list]], Y[[stu - 1 for stu in test_stu_list]]
        return A, Y[[stu - 1 for stu in test_stu_list]]


if __name__ == '__main__':
    from initial_dataSet2 import DataSet

    # ----------基本参数--------------
    basedir = './'
    dataSet_list = ('FrcSub', 'ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC')
    epochs_list = (8, 8, 10, 1, 2)

    dataSet_idx = 1
    test_ratio = 0.2
    batch_size = 64
    learn_rate = 3e-2
    n_splits = 3

    data_set_name = dataSet_list[dataSet_idx]
    epochs = epochs_list[dataSet_idx]
    device = 'cuda'
    # ----------基本参数--------------

    dataSet = DataSet(basedir, data_set_name)
    train_data, test_data = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
    exer_conc_adj = dataSet.get_exer_conc_adj()

    total_stu_list = dataSet.total_stu_list

    model = CICDM(student_num=dataSet.student_num,
                  concept_num=dataSet.concept_num,
                  exercise_num=dataSet.exercise_num,
                  exer_conc_adj=exer_conc_adj,
                  lr=learn_rate,
                  device=device)

    index_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                              batch_size=batch_size, shuffle=True)

    model.fit(index_loader, train_data, epochs=epochs, n_splits=n_splits, test_df=test_data)
    # acc, auc, rmse, mae = model.test(index_loader, train_data, test_data)
    cognitive_state, score_pred = model.get_A_and_Y(index_loader, dataSet.record)
