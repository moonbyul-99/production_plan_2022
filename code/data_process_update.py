import numpy as np
import pandas as pd
import loss_constraint as lc
import algo_update as al 
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


def process(data_path,res_path,length,file_name,N,T,alpha,beta,use_seed,seed,cross_type,mutation_type,p_mut):
# 单文件处理
#args 元组 记录数据地址及遗传算法参数
#data_path: 需要处理的文件地址
#length 当前问题规模
#res_path:  生成的结果的文件夹地址
#file_name: 当前处理的文件名称

#------------------------------------------------------------
#处理当前数据
    data = pd.read_csv(data_path)
    data = data.drop(columns= ['Unnamed: 0'],axis = 1)
    data['车顶颜色'][data['车顶颜色'] == '无对比颜色'] = data['车身颜色'][data['车顶颜色'] == '无对比颜色']


    print(file_name.split('.')[0] + '  群体数量' + str(N) + '   迭代次数' + str(T))
    population, mean_record, best_record = al.GA(data,N,T,alpha,beta,use_seed,seed,cross_type, mutation_type,p_mut)
#--------------------------------
#可视化目标函数变化并保存
#generate random data
    fig, axs = plt.subplots(1,3,figsize=(30,10))
    y1 = best_record[:,0]
    y2 = mean_record[:,0]
    x = [ i for i in range(1,T+1)]
    axs[0].plot(x,y1,x,y2)
    axs[0].set_title('loss 1')

    y1 = best_record[:,1]
    y2 = mean_record[:,1]
    axs[1].plot(x,y1,x,y2)
    axs[1].set_title('loss 2')

    y1 = best_record[:,3]
    y2 = mean_record[:,3]
    axs[2].plot(x,y1,x,y2)
    axs[2].set_title('loss total')

    tmp_path = os.path.join(res_path,file_name.split('.')[0])
    plt.savefig(tmp_path)
    plt.clf()
#------------------------------
#将结果保存至指定的excel中
    #保留总用时最少的样本
    #cost1 12  cost2 12 cost_all  26
    best_population = np.zeros((50,length),dtype = np.int32)
    cost_1 = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)[:,0].reshape(N)
    sel_id = np.argpartition(np.array(cost_1), 12)[0:12]
    best_population[0:12,:]= population[sel_id,:]

    cost_2 = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)[:,1].reshape(N)
    sel_id = np.argpartition(np.array(cost_2), 12)[0:12]
    best_population[12:24,:]= population[sel_id,:]

    cost_all = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)[:,3].reshape(N)
    sel_id = np.argpartition(np.array(cost_1), 26)[0:26]
    best_population[24:,:]= population[sel_id,:]

    col = []
    for j in range(1,length+1):
        col.append('Variable ' + str(j))
    df = pd.DataFrame(best_population,columns= col)
    df.to_excel(os.path.join(res_path,file_name.split('.')[0] + '.xlsx'),index=False)
    # 记录最终生成的群体
    df_2 = pd.DataFrame(population)
    df_2.to_csv(os.path.join(res_path,file_name.split('.')[0] + '.csv'),index=False)