import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
import  gc

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
    def __init__(self, basedir, dataSetName, build=False):
        self.basedir = basedir
        self.dataSetName = dataSetName
        if dataSetName == 'FrcSub':
            read_dir = basedir + 'data/frcSub/'
            save_result_dir = basedir + 'output/result/frcSub/'
            save_parameter_dir = basedir + '/output/parameter/frcSub/'
            N = 536
            J = 20
            K = 8
        elif dataSetName == 'Math1':
            read_dir = basedir + 'data/math1/'
            save_result_dir = basedir + 'output/result/math1/'
            save_parameter_dir = basedir + '/output/parameter/math1/'
            N = 4209
            J = 20
            K = 11
        elif dataSetName == 'Math2':
            read_dir = basedir + 'data/math2/'
            save_result_dir = basedir + 'output/result/math2/'
            save_parameter_dir = basedir + '/output/parameter/math2/'
            N = 3911
            J = 20
            K = 16
        elif dataSetName == 'ASSIST_0910':
            read_dir = basedir + 'data/a0910/'
            save_result_dir = basedir + 'output/result/a0910/'
            save_parameter_dir = basedir + '/output/parameter/a0910/'
            # N = 2392
            # J = 17657
            # K = 123
            N = 2380
            J = 16804
            K = 110
        elif dataSetName == 'ASSIST_2017':
            read_dir = basedir + 'data/a2017/'
            save_result_dir = basedir + 'output/result/a2017/'
            save_parameter_dir = basedir + '/output/parameter/a2017/'
            N = 1678
            J = 2210
            K = 101
        elif dataSetName == 'KDDCUP':
            read_dir = basedir + 'data/kddcup/'
            save_result_dir = basedir + 'output/result/kddcup/'
            save_parameter_dir = basedir + '/output/parameter/kddcup/'
            N = 425
            J = 101744
            K = 16
        elif dataSetName == 'MAT_2016':
            read_dir = basedir + 'data/mat2016/'
            save_result_dir = basedir + 'output/result/mat2016/'
            save_parameter_dir = basedir + '/output/parameter/mat2016/'
            N = 6866
            J = 1847
            K = 445
        elif dataSetName == 'JUNYI':
            read_dir = basedir + 'data/junyi/'
            save_result_dir = basedir + 'output/result/junyi/'
            save_parameter_dir = basedir + '/output/parameter/junyi/'
            N = 36591
            J = 721
            K = 721
        elif dataSetName == 'MathEC':
            read_dir = basedir + 'data/math_ec/'
            save_result_dir = basedir + 'output/result/math_ec/'
            save_parameter_dir = basedir + '/output/parameter/math_ec/'
            N = 118971
            J = 27613
            K = 388
        else:
            print('Dataset does not exist!')
            exit(0)
        print('DataSet:', dataSetName)
        item = pd.read_csv(read_dir + "item.csv").set_index('item_id')
        data = pd.read_csv(read_dir + "record.csv").set_index('user_id')
        # if not build:
        #     conc_relation = pd.read_csv(read_dir + "concept_relationship.csv")
        #     self.conc_relation = conc_relation
        if dataSetName != "FrcSub":
            if not build:
                conc_relation = pd.read_csv(read_dir + "concept_relationship.csv")
                self.conc_relation = conc_relation

        self.total_stu_list = np.unique(data.index)
        self.student_num = N
        self.exercise_num = J
        self.concept_num = K
        self.record = data
        self.item = item

        self.read_dir = read_dir
        self.save_result_dir = save_result_dir
        self.save_parameter_dir = save_parameter_dir

    def get_train_test(self, record, test_ratio=0.2):
        print('test_ratio:', test_ratio)
        # record = record[0:int(len(record)*0.1)]
        train, test_train, test_valid, train_stu_list, test_stu_list,test_record = split_record2(record, test_ratio=test_ratio)
        return train, test_train, test_valid, train_stu_list, test_stu_list,test_record

    def get_item_concept_df(self) -> pd.DataFrame:
        item = self.item #item : item_id , knowledge_code
        item = item[~item.index.duplicated()]  # 去除重复项
        item_list, concept_list = [], []
        for idx in item.index:
            now_concept_list = eval(item.loc[idx, 'knowledge_code'])
            item_list.extend([idx] * len(now_concept_list))
            concept_list.extend(now_concept_list)
        item_conc_idx = range(len(concept_list))
        return pd.DataFrame({'item': item_list, 'concept': concept_list, 'item_conc_index': item_conc_idx}).astype('int')

    def get_exer_conc_adj(self): #计算Q
        Q = np.zeros((self.exercise_num, self.concept_num), dtype='bool')
        item = self.item #item : item_id , knowledge_code
        item = item[~item.index.duplicated()] # 去除重复值
        for idx in item.index:
            know_list = eval(item.loc[idx, 'knowledge_code'])  # eval 函数可将数值字符串转换为数值
            Q[np.array([idx] * len(know_list)) - 1, np.array(know_list) - 1] = True
        return torch.tensor(Q, dtype=torch.float)

    def get_conc_conc_adj(self): #conc_relation csv 源数据集无
        conc_graph = np.zeros((self.concept_num, self.concept_num), dtype='bool')
        conc_graph[self.conc_relation['parent'] - 1, self.conc_relation['knowledge_code'] - 1] = True
        return torch.tensor(conc_graph, dtype=torch.float)
