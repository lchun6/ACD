#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: Bean
# datetime:2023/1/11 14:31
import sys
sys.path.append("../")
from initial_dataSet2 import DataSet
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


basedir = '../'
models=['SCD','CDGK','NCD','DINA','QRCDM','ICD']

dataSet_list = ('ASSIST_2017','JUNYI', 'MathEC')
save_list = ('a2017/', 'junyi/', 'math_ec/')

shuliandu = pd.DataFrame(columns=['model', 'ASSIST_2017','JUNYI', 'MathEC'])
defen = pd.DataFrame(columns=['model', 'ASSIST_2017','JUNYI', 'MathEC'])
for model_idx in range(len(models)):
    # model_idx=4
    # model_idx=2
    model=models[model_idx]
    shuliandu.loc[model_idx,'model'] = model
    defen.loc[model_idx,'model'] = model
    for dataSet_idx in range(len(dataSet_list)):
        # dataSet_idx=4
        data_set_name = dataSet_list[dataSet_idx]
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

        # cogn_state=np.loadtxt(read_dir+'cognitive_state.csv',delimiter=',')
        cogn_state=np.loadtxt(read_dir+'new_stu_cogn_state.csv',delimiter=',')
        # record=dataSet.record.reset_index()
        record=pd.read_csv(read_dir+'test_record.csv', names=['user_id', 'item_id','score'] )

        cluster_k=4

        cluster=KMeans(n_clusters=cluster_k)
        cluster.fit(cogn_state)
        labels=cluster.labels_
        user_id = record['user_id'].unique()
        df = pd.DataFrame({'user_id': user_id , 'new_user_id': [x+1 for x in range(len(user_id))]})
        record = pd.merge(record, df, on='user_id')
        record = record.drop('user_id', axis=1)
        record.rename(columns={'new_user_id': 'user_id'}, inplace=True)
        # record['cluster']=labels[record['user_id'].astype(int)-1]
        record['cluster']=labels[record['user_id'].astype(int)-1]
        item_mean_score=np.ones((cluster_k,dataSet.exercise_num))*-1e10
        for cluster_num in range(cluster_k):
            clust_record=record.set_index('cluster').loc[cluster_num,:]
            mean_score=clust_record.groupby('item_id').mean()['score']
            item_mean_score[cluster_num,mean_score.index.astype(int)-1]=mean_score.values
        item_mean_score[item_mean_score<-1e5]=np.nan

        centers=cluster.cluster_centers_
        cent_abs_list=[]
        for i in range(cluster_k):
            for j in range(i+1,cluster_k):
                diff=centers[i]-centers[j]
                diff=diff[~np.isnan(diff)]
                cent_abs_list.append(np.abs(diff).mean())

        print(model,data_set_name)
        print('熟练度区分度:',np.mean(cent_abs_list)/record['score'].mean())
        shuliandu.loc[model_idx,data_set_name] = np.mean(cent_abs_list)/record['score'].mean()

        abs_list=[]
        for i in range(cluster_k):
            for j in range(i+1,cluster_k):
                diff=item_mean_score[i]-item_mean_score[j]
                diff=diff[~np.isnan(diff)]
                abs_list.append(np.abs(diff).mean())

        print(model,data_set_name)
        print('得分区分度:',np.mean(abs_list)/record['score'].mean())
        defen.loc[model_idx, data_set_name] = np.mean(abs_list)/record['score'].mean()
shuliandu.set_index('model',inplace=True)
defen.set_index('model',inplace=True)

shuliandu.to_csv('熟练度区分度.csv')
defen.to_csv('得分区分度.csv')
