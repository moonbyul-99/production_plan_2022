import numpy as np
import pandas as pd
import loss_constraint as lc
import algorithm as al 
import matplotlib.pyplot as plt 
import os
from os import walk


def path_process(path):
    # 设置路径
    #path = r'D:\sun_python\PROJECT\生产调度\100组数据集\数据集'
    os.chdir(path)

    # 遍历文件
    res = []
    for (dir_path, dir_name, file_name) in walk(path,topdown=True):
        res.extend(file_name)


    length = np.array([eval(i.split('_')[1].split('.')[0]) for i in res])
    id_sort = np.argsort(length)
    file_name_list = []
    for i in id_sort:
        file_name_list.append(res[i])

    length = length[id_sort]


    # 拼接全路径
    final_path = []
    for file_name in file_name_list:
        tmp_final_path = os.path.join(path, file_name)
        final_path.append(tmp_final_path)

    return length, file_name_list, final_path
# length 记录了每个子问题的大小
# res_new 按照子问题大小规模进行的文件排序
# final_path 子问题文件路径列表

def process(data_path,res_path,length,file_name,use_seed = False, seed_parent = None):
# 单文件处理
#data_path: 需要处理的文件地址
#res_path:  生成的结果的文件夹地址
#seed_parent: 之前优化的结果
#file_name: 当前处理的文件名称
#当前的问题规模


#------------------------------------------------------------
#处理当前数据
    data = pd.read_csv(data_path)
    data = data.drop(columns= ['Unnamed: 0'],axis = 1)
    data['车顶颜色'][data['车顶颜色'] == '无对比颜色'] = data['车身颜色'][data['车顶颜色'] == '无对比颜色']

    #if use_seed:
    #    seed_parent = pd.read_excel(seed_parent,sheet_name= file_name_list[case_id].split('.')[0] )
    #else:
    #    seed_parent = None
    #设置群体数目和迭代次数
    N = min(400 + 10*int(length/10),1000)
    T = min(600 + 20*int(length/10),5000)
    #N = 100
    #T = 30
    print(file_name.split('.')[0] + '  群体数量' + str(N) + '   迭代次数' + str(T))
    population, mean_cost, best_cost, mean_record, best_record = al.GA(data,N,T,0.1,0.4,use_seed,seed_parent)

#--------------------------------
#可视化目标函数变化并保存
    y1 = best_record
    y2 = mean_record
    x = [ i for i in range(1,T+1)]

    #l1=plt.plot(x,y1,label='type1')
    #l2=plt.plot(x,y2,label='type2')

    plt.plot(x,y1,x,y2)
    plt.title('loss iteration for '+ file_name.split('.')[0])
    plt.xlabel('iteration number')
    plt.ylabel('loss')
    tmp_path = os.path.join(res_path,file_name.split('.')[0])
    plt.savefig(tmp_path)
    plt.clf()
#------------------------------
#将结果保存至指定的excel中
    #保留总用时最少的50个样本
    cost = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)[:,3].reshape(N)
    sel_id = np.argpartition(np.array(cost), 50)[0:50]
    best_population = population[sel_id,:]

    col = []
    for j in range(1,length+1):
        col.append('Variable ' + str(j))
    df = pd.DataFrame(best_population,columns= col)
    df.to_excel(os.path.join(res_path,file_name.split('.')[0] + '.xlsx'),index=False)

def batch_process(data_path,res_path,length,file_name_list,use_seed = False, seed_parent_path = None):
    # 批量处理数据
    #data_path: 列表 保存需要处理的文件的文件地址
    #res_path:  生成的结果的文件夹地址
    #seed_parent: 之前优化的结果的保存路径
    writer = pd.ExcelWriter(res_path)
    for case_id in range(0,len(data_path)):
    #------------------------------------------------------------
    #依次处理生产计划
        path = data_path[case_id]
        data = pd.read_csv(path)
        data = data.drop(columns= ['Unnamed: 0'],axis = 1)
        data['车顶颜色'][data['车顶颜色'] == '无对比颜色'] = data['车身颜色'][data['车顶颜色'] == '无对比颜色']

        if use_seed:
            seed_parent = pd.read_excel(seed_parent_path,sheet_name= file_name_list[case_id].split('.')[0] )
        else:
            seed_parent = None
        #设置群体数目和迭代次数
        #N = min(400 + 10*int(length[case_id]/10),1000)
        #T = min(600 + 20*int(length[case_id]/10),5000)
        N = 100
        T = 60
        print(file_name_list[case_id].split('.')[0] + '  群体数量' + str(N) + '   迭代次数' + str(T))
        population, mean_cost, best_cost, mean_record, best_record = al.GA(data,N,T,0.1,0.4,use_seed,seed_parent)

    #--------------------------------
    #可视化目标函数变化并保存
        y1 = best_record
        y2 = mean_record
        x = [ i for i in range(1,T+1)]

        #l1=plt.plot(x,y1,label='type1')
        #l2=plt.plot(x,y2,label='type2')

        plt.plot(x,y1,x,y2)
        plt.title('loss iteration for '+ file_name_list[case_id].split('.')[0])
        plt.xlabel('iteration number')
        plt.ylabel('loss')
        tmp_path = os.path.join(os.path.abspath(os.path.dirname(res_path)),file_name_list[case_id].split('.')[0])
        plt.savefig(tmp_path)
        plt.clf()
    #------------------------------
    #将结果保存至指定的excel中
        #保留总用时最少的50个样本
        cost = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)[:,3].reshape(N)
        sel_id = np.argpartition(np.array(cost), 50)[0:50]
        best_population = population[sel_id,:]



        col = []
        for j in range(1,length[case_id]+1):
            col.append('Variable ' + str(j))
        df = pd.DataFrame(best_population,columns= col)
        df.to_excel(writer,file_name_list[case_id].split('.')[0],index=False)
    writer.save()
