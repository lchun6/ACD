import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm

def split_record2(record, test_ratio=0.2):
    '''
    record:userid itemid score
    '''
    total_stu_list =list(set(record.index)) #index集合
    n_total = len(total_stu_list)
    offset = int(n_total * test_ratio)
    random.shuffle(total_stu_list)
    test_stu_list = total_stu_list[:offset]
    train_stu_list = total_stu_list[offset:]
    train_data = [[], [], []]
    test_train_data = [[], [], []]
    test_valid_data = [[], [], []]
    test_record = record.loc[test_stu_list, :]
    for stu in tqdm(train_stu_list, "[split train record:]"): #[split record:] 进度条显示的名字 根据userid遍历 把每个user的答题记录按照比例分为训练集和测试集
        stu_data = record.loc[stu, :]  # record:userid itemid score 一个学生回答多个问题
        stu_item = np.array(stu_data['item_id']) # item_id
        stu_score = np.array(stu_data['score']) # score

        length = len(stu_item) #答题个数len
        index_list = list(range(length)) #0,1,2...len

        train_data[0].extend([stu] * len(index_list))
        train_data[1].extend(stu_item[index_list])
        train_data[2].extend(stu_score[index_list])
    train = pd.DataFrame({'user_id': train_data[0], 'item_id': train_data[1], 'score': train_data[2]}).set_index('user_id')

    for stu in tqdm(test_stu_list , "[split test record:]"):  # [split record:] 进度条显示的名字 根据userid遍历 把每个user的答题记录按照比例分为训练集和测试集
        stu_data = record.loc[stu, :]  # record:userid itemid score 一个学生回答多个问题
        stu_item = np.array(stu_data['item_id'])  # item_id
        stu_score = np.array(stu_data['score'])  # score
        # test_record= stu_data
        length = len(stu_item)  # 答题个数len
        index_list = list(range(length))  # 0,1,2...len
        test_index = random.sample(index_list, int(length * test_ratio))  # 在index_list中选取len*0.2个index
        train_index = list(set(index_list) - set(test_index))

        test_train_data[0].extend([stu] * len(train_index))
        test_train_data[1].extend(stu_item[train_index])
        test_train_data[2].extend(stu_score[train_index])

        test_valid_data[0].extend([stu] * len(index_list))
        test_valid_data[1].extend(stu_item[index_list])
        test_valid_data[2].extend(stu_score[index_list])


    test_train = pd.DataFrame({'user_id': test_train_data[0], 'item_id': test_train_data[1], 'score': test_train_data[2]}).set_index('user_id')
    test_valid = pd.DataFrame({'user_id': test_valid_data[0], 'item_id': test_valid_data[1], 'score': test_valid_data[2]}).set_index('user_id')

    return train,test_train,test_valid,train_stu_list,test_stu_list,test_record

class DataSet():
    def __init__(self, basedir, dataSetName):
        self.basedir = basedir
        self.dataSetName = dataSetName
        if dataSetName == 'FrcSub':
            read_dir = basedir + '/data/frcSub/'
            save_dir = basedir + '/output/frcSub/'
            n = 536
            m = 20
            k = 8
        elif dataSetName == 'Math1':
            read_dir = basedir + '/data/math1/'
            save_dir = basedir + '/output/math1/'
            n = 4209
            m = 20
            k = 11
        elif dataSetName == 'Math2':
            read_dir = basedir + '/data/math2/'
            save_dir = basedir + '/output/math2/'
            n = 3911
            m = 20
            k = 16
        elif dataSetName == 'ASSIST_0910':
            read_dir = basedir + '/data/a0910/'
            save_dir = basedir + '/output/a0910/'
            n = 4163
            m = 17746
            k = 123
        elif dataSetName == 'ASSIST_2017':
            read_dir = basedir + '/data/a2017/'
            save_dir = basedir + '/output/a2017/'
            n = 1678
            m = 2210
            k = 101
        else:
            print('Dataset does not exist!')
            exit(0)
        print('数据集：', dataSetName)
        item = pd.read_csv(read_dir + "item.csv")

        train_data = pd.read_csv(read_dir + "train.csv").set_index('user_id')
        test_data = pd.read_csv(read_dir + "test.csv").set_index('user_id')

        if dataSetName in ('FrcSub', 'ASSIST_0910', 'ASSIST_2017'):
            if dataSetName != 'FrcSub':
                valid_data = pd.read_csv(
                    read_dir + "valid.csv").set_index('user_id')
            else:
                valid_data = pd.read_csv(
                    read_dir + "test.csv").set_index('user_id')
            obj_prob_index = 'All'
            sub_prob_index = None
        else:
            valid_data = pd.read_csv(read_dir + "test.csv").set_index('user_id')
            # type of problems
            obj_prob_index = np.loadtxt(
                read_dir + "obj_prob_index.csv", delimiter=',', dtype=int)
            sub_prob_index = np.loadtxt(
                read_dir + "sub_prob_index.csv", delimiter=',', dtype=int)

        self.total_stu_list = set(train_data.index) & \
            set(valid_data.index) & set(test_data.index)

        self.stu_num = n
        self.prob_num = m
        self.skill_num = k
        self.item = item
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.save_dir = save_dir
        self.obj_prob_index = obj_prob_index
        self.sub_prob_index = sub_prob_index


    def get_train_test(self, record, test_ratio=0.2):
        print('test_ratio:', test_ratio)
        # record = record[0:int(len(record)*0.1)]
        train, test_train, test_valid, train_stu_list, test_stu_list,test_record = split_record2(record, test_ratio=test_ratio)
        return train, test_train, test_valid, train_stu_list, test_stu_list,test_record


    def get_Q(self):
        Q = np.zeros((self.prob_num, self.skill_num), dtype='bool')
        item = self.item
        for idx in item.index:
            item_id = item.loc[idx, 'item_id']
            know_list = item.loc[idx, 'knowledge_code'].replace(
                '[', '').replace(']', '').split(',')
            for know in know_list:
                Q[item_id - 1, int(know) - 1] = True
        return torch.tensor(Q, dtype=torch.float)
