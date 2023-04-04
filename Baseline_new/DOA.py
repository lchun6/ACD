#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: Bean
# datetime:2023/1/10 17:17
import sys
sys.path.append("../")
from initial_dataSet2 import DataSet
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

basedir = '../'
baseline_list=('DINA','NCD','CDGK')
baseline=baseline_list[0]

dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC', 'FrcSub')
save_list = ('a0910/', 'a2017/', 'junyi/', 'math_ec/', 'frcsub/')

dataSet_idx=0
data_set_name = dataSet_list[dataSet_idx]
dataSet = DataSet(basedir, data_set_name)

def get_item_concept_df(item:pd.DataFrame) -> pd.DataFrame:
    item = item[~item.index.duplicated()]  # 去除重复项
    item_list, concept_list = [], []
    for idx in item.index:
        now_concept_list = eval(item.loc[idx, 'knowledge_code'])
        item_list.extend([idx] * len(now_concept_list))
        concept_list.extend(now_concept_list)
    return pd.DataFrame({'item': item_list, 'concept': concept_list}).astype('int')

read_dir='./'+baseline+'/output/'+save_list[dataSet_idx]

cogn_state=np.loadtxt(read_dir+'cognitive_state.csv',delimiter=',')
record = pd.read_csv(read_dir+'test_record.csv', names=['user_id', 'item_id','score'] )
# record=dataSet.record.reset_index()
item_conc=get_item_concept_df(dataSet.item)

doa_list=[]
for k in tqdm(range(1,cogn_state.shape[1]+1)):
    cong_doa_count=0
    item_cong_doa_count=0
    item_k=item_conc[item_conc['concept']==k]['item'].unique()
    item_k_ = set(item_k) & set(record.set_index('item_id').index)
    record_k=record.set_index('item_id').loc[item_k_,:]  #有的习题在习题记录里面没有 需要先删除掉 不然索引不存在会报错
    if len(item_k_)>0 and len(record_k)>0:
        for j in item_k_:
            record_j=record_k.loc[j,:]
            stu_j=np.array(record_j['user_id']).reshape(-1).astype('int')
            cong_j=cogn_state[stu_j-1,k-1]
            sort_stu_idx=cong_j.argsort() # 从小到大
            sort_score_j=np.array(record_j['score']).reshape(-1)[sort_stu_idx]
            sort_cong_j=cong_j[sort_stu_idx]
            for i in range(len(sort_stu_idx)):
                fliter=((sort_cong_j[i:]-sort_cong_j[i])>=0)
                cong_doa_count+=fliter.sum()
                score_i=sort_score_j[i:][fliter]
                if len(score_i)>0:
                    item_cong_doa_count+=((score_i-sort_score_j[i])>=0).sum()
        if cong_doa_count>0:
            doa_list.append(item_cong_doa_count/cong_doa_count)

print(baseline)
print('{} 的DOA = {}'.format(data_set_name,np.mean(doa_list)))