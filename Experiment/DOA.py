#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: Bean
# datetime:2023/1/7 22:03
import sys
sys.path.append("../")
from initial_dataSet2 import DataSet
import pandas as pd
import numpy as np
from tqdm import tqdm

basedir = '../'
DOA = pd.DataFrame(columns=['model', 'ASSIST_2017','JUNYI', 'MathEC'])

models=['SCD','NCD','DINA','QRCDM','ICD']
dataSet_list = ( 'ASSIST_2017','JUNYI', 'MathEC')
save_list = ('a2017/', 'junyi/', 'math_ec/')
for model_idx in range(len(models)):
    # model_idx=2
    
    model=models[model_idx]
    print('model:'+model)
    DOA.loc[model_idx,'model'] = model
    for dataSet_idx,data_set_name in enumerate(dataSet_list):

        dataSet = DataSet(basedir, data_set_name,build=True)
        if model=='SCD':
            read_dir=dataSet.save_result_dir
        elif model=='CDGK':
            read_dir='../Baseline_new/CDGK/output/'+save_list[dataSet_idx]
        elif model=='NCD':
            read_dir='../Baseline_new/NCD/output/'+save_list[dataSet_idx]
        elif model=='DINA':
            read_dir='../Baseline_new/DINA/output/'+save_list[dataSet_idx]
        elif model=='QRCDM':
            read_dir='../Baseline_new/QRCDM/output/'+save_list[dataSet_idx]
        elif model=='ICD':
            read_dir='../Baseline_new/ICD/output/'+save_list[dataSet_idx]
        else:
            assert False,'模型名称错误'

        def get_item_concept_df(item) -> pd.DataFrame:
            item = item[~item.index.duplicated()]  # 去除重复项
            item_list, concept_list = [], []
            for idx in item.index:
                now_concept_list = eval(item.loc[idx, 'knowledge_code'])
                item_list.extend([idx] * len(now_concept_list))
                concept_list.extend(now_concept_list)
            return pd.DataFrame({'item': item_list, 'concept': concept_list}).astype('int')

        # read_dir=dataSet.save_result_dir

        
        cogn_state=np.loadtxt(read_dir+'cognitive_state.csv',delimiter=',')
        record = pd.read_csv(read_dir+'test_record.csv', names=['user_id', 'item_id','score'] )

        item_conc=get_item_concept_df(dataSet.item)  #习题和知识点矩阵

        def get_DOA(record,cogn_state,item_conc):
            doa_list=[]
            unique_item=record['item_id'].unique()
            for k in tqdm(range(1,cogn_state.shape[1]+1)):
                cong_doa_count=0
                item_cong_doa_count=0
                item_k=item_conc[item_conc['concept']==k-1]['item'].unique()
                item_list=list(set(unique_item)&set(item_k))
                record_k=record.set_index('item_id').loc[item_list,:]
                if len(item_list)>0 and len(record_k)>0:
                    for j in item_list:
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
            return np.mean(doa_list)

        doa=get_DOA(record,cogn_state,item_conc)
        DOA.loc[model_idx,data_set_name] = doa
        print('{} 的DOA = {}'.format(data_set_name,doa))
DOA.set_index('model',inplace=True)
DOA.to_csv('DOA3.csv')