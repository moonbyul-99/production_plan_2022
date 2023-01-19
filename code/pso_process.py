import numpy as np
import pandas as pd
import loss_constraint as lc
import pso
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


def process(data_path,res_path,length,file_name,N,r0,r1,T,lb,ub,alpha,c1,c2,c,SA_T,K,anneal):
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
    p_best,g_best_record = pso.PSO(data,N,r0,r1,T,lb,ub,alpha,c1,c2,c,SA_T,K,anneal)
#--------------------------------
#可视化目标函数变化并保存
    y = g_best_record
    x = [ i for i in range(1,T+1)]

    plt.plot(x,y)
    plt.title('loss iteration for '+ file_name.split('.')[0])
    plt.xlabel('iteration number')
    plt.ylabel('loss')
    tmp_path = os.path.join(res_path,file_name.split('.')[0])
    plt.savefig(tmp_path)
    plt.clf()
#------------------------------
#将结果保存至指定的excel中
    #保留总用时最少的样本
    if p_best.shape[0] > 50:
        cost = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = p_best)[:,3].reshape(N)
        sel_id = np.argpartition(np.array(cost), 50)[0:50]
        best_population = p_best[sel_id,:]
    else:
        best_population = p_best

    col = []
    for j in range(1,length+1):
        col.append('Variable ' + str(j))
    df = pd.DataFrame(best_population,columns= col)
    df.to_excel(os.path.join(res_path,file_name.split('.')[0] + '.xlsx'),index=False)

