# 算法
import numpy as np
import pandas as pd
import loss_constraint as lc

#-----------------------------------------
#遗传算法

#生成满足约束的一组解

def generator(data,N):
    n = data.shape[0]
#  使用numpy 操作代替list 提高速度
    population = np.apply_along_axis(lambda x: np.random.permutation(n),axis=1,arr =np.zeros((N,n)))
    return population

#--------------------------------------------------------------------------------
#交叉算子  基于两个亲本生成子代
def cross_op_1(a,b):
# 对a,b 某个节点之后的元素进行交叉，节点之前的保持不变
    child_a = np.copy(a)
    child_b = np.copy(b)
    n = a.shape[0]
    point = np.random.randint(1,n-1)
    part_a = a[point:]
    part_b = b[point:]
    #记录a中后面部分的排序
    id_a = []
    #记录b中后面部分的排序
    id_b = []
    for i in range(0,n-point):
        id_a.append(np.where(b == part_a[i])[0][0])
        id_b.append(np.where(a == part_b[i])[0][0])
    id_a.sort()
    id_b.sort()
    child_a[point:] = b[id_a]
    child_b[point:] = a[id_b]
    return child_a,child_b

def cross_op_2(a,b):
# 对a,b 某个节点之前的元素进行交叉，节点之后的保持不变
    child_a = np.copy(a)
    child_b = np.copy(b)
    n = a.shape[0]
    point = np.random.randint(1,n-1)
    part_a = a[0:point]
    part_b = b[0:point]
    #记录a中后面部分的排序
    id_a = []
    #记录b中后面部分的排序
    id_b = []
    for i in range(0,point):
        id_a.append(np.where(b == part_a[i])[0][0])
        id_b.append(np.where(a == part_b[i])[0][0])
    id_a.sort()
    id_b.sort()
    child_a[0:point] = b[id_a]
    child_b[0:point] = a[id_b]
    return child_a,child_b


#-----------------------------------------
# 变异算子  基于单个亲本进行更新
def swap(parent):
    # 交换亲本中两个元素位置
    n = parent.shape[0]
    x = np.array([i for i in range(0,n)])
    cut = np.sort(np.random.choice(x,2,replace= False))
    child = np.copy(parent)
    child[cut[0]] = parent[cut[1]]
    child[cut[1]] = parent[cut[0]]
    return child

def reverse(parent):
    #将亲本中间的片段进行倒序
    n = parent.shape[0]
    x = np.array([i for i in range(0,n+1)])
    cut = np.sort(np.random.choice(x,2,replace= False))
    child = np.copy(parent)
    child[cut[0]:cut[1]] = np.flipud(parent[cut[0]:cut[1]])
    return child

def exchange_1(parent):
    #将一条序列截断为三段，123-> 231
    n = parent.shape[0]
    x = np.array([i for i in range(0,n-1)])
    cut = np.sort(np.random.choice(x,2,replace= False))
    child = np.zeros(n,dtype = int)
    child[0:cut[1] - cut[0]] = parent[cut[0]:cut[1]] 
    child[cut[1] - cut[0]:n-cut[0]]  =  parent [cut[1]:] 
    child[n-cut[0]:] =  parent[0:cut[0]]
    return child

def exchange_2(parent):
    #将一条序列截断为三段，123-> 321
    n = parent.shape[0]
    x = np.array([i for i in range(0,n-1)])
    cut = np.sort(np.random.choice(x,2,replace= False))
    child = np.zeros(n,dtype = int)
    child[0:n - cut[1]] = parent[cut[1]:] 
    child[n-cut[1]:n-cut[0]]  =  parent [cut[0]:cut[1]] 
    child[n-cut[0]:] =  parent[0:cut[0]]
    return child

def auto_mutation(parent):
    #随机选择变异方式
    id = np.random.choice([0,1,2,3])
    if id == 0:
        child = swap(parent)
    elif id == 1:
        child = reverse(parent)
    elif id == 2:
        child = exchange_1(parent)
    else:
        child = exchange_2(parent)
    return parent

#---------------------------------------------------------
#遗传算法主程序

def GA(data,N,T,alpha,beta,use_seed = False, seed = None,cross_type = 'cross_op_1', mutation_type = 'auto_mutation',p_mut = 0.8):
    #N 为群体数目  T为迭代次数  alpha  为群体中直接保留的样本比例  beta 为群体中生成后代的样本比例 alpha + beta <= 1
    #population 初始群体 N个样本
    #cross_type  交叉算子类型 cross_op_1  cross_op_2
    #mutation_type 变异算子类型  auto_mutation swap reverse exchange_1 exchange_2
    #p_mut  后代变异概率
    #多目标优化，基于三个优化目标生成后代
    if use_seed == False:
        population = generator(data,N)
    else:
        population = seed

    n_1 = 2*int(alpha*N/2)
    n_2 = 2*int(beta*N/2)
    n_3 = N - n_1 - n_2
    cost = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)
    
    best_cost = np.min(cost,axis = 0)
    mean_record = np.zeros((T,4),dtype = np.int32)
    best_record = np.zeros((T,4),dtype = np.int32)


    for t in range(0,T):
        cost = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)

        # 计算样本的适应度  scale(1/cost)
        fitness = np.divide(1,cost)
        fitness = fitness/np.sum(fitness,axis = 0)  #用时较少的方案有较大的适应度

        # 更新不同目标的最优值
        x = np.min(cost,axis = 0)
        if x[0] < best_cost[0]:
            best_cost[0] = x[0]
        if x[1] < best_cost[1]:
            best_cost[1] = x[1]
        if x[3] < best_cost[3]:
            best_cost[3] = x[3]

        mean_record[t,:] = np.mean(cost,axis = 0)
        best_record[t,:] = best_cost
        

        # 保留不同优化目标下的前n1个最佳样本 默认使用 0.3*n1 0.3*n1 0.4*n1的规模保存最佳样本
        best_population = np.zeros((n_1,data.shape[0]))
        a = int(0.3*n_1)
        b = n_1 - 2*a
        sel_id_0 = np.argpartition(np.array(cost[:,0]), a)[0:a]
        best_population[0:a,:] = population[sel_id_0,:]
        sel_id_1 = np.argpartition(np.array(cost[:,1]), a)[0:a]
        best_population[a:2*a,:] = population[sel_id_1,:]
        sel_id_3 = np.argpartition(np.array(cost[:,3]), b)[0:b]
        best_population[2*a:n_1,:] = population[sel_id_3,:]        

        # 通过交叉算子生成不同优化目标下的n2个子代， 0.3 0.3 0.4
        # 群体中生成n2个新的样本
        children = get_child(population, fitness, data, n_2)

        # 通过变异算子对子代进行变异
        mut_id = np.random.rand(n_2) >= p_mut
        if mutation_type == 'auto_mutation':
            children[mut_id,:] = np.apply_along_axis(lambda x: auto_mutation(x),axis=1,arr = children[mut_id,:])
        elif mutation_type == 'swap':
            children[mut_id,:]  = np.apply_along_axis(lambda x: swap(x),axis=1,arr = children[mut_id,:])
        elif mutation_type == 'reverse':
            children[mut_id,:]  = np.apply_along_axis(lambda x: reverse(x),axis=1,arr = children[mut_id,:])
        elif mutation_type == 'exchange_1':
            children[mut_id,:]  = np.apply_along_axis(lambda x: exchange_1(x),axis=1,arr = children[mut_id,:])
        elif mutation_type == 'exchange_2':
            children[mut_id,:]  = np.apply_along_axis(lambda x: exchange_2(x),axis=1,arr = children[mut_id,:])
        
        # 群体中引入n3个全新的样本
        new_population = generator(data, n_3)

        
        if t%50 == 0:
            print(str(t) + '  iteration' + '     best loss' + str(best_cost))

        #population = best_population + children + new_population
        population[0:n_1,:] = best_population
        population[n_1:n_1 +n_2,:] = children
        population[n_1+n_2:N,:] = new_population
    
    return population, mean_record, best_record


#从群体中生成一组新的样本
def get_child(population, fitness, data, n_2, cross_type = 'cross_op_1'):
    N = population.shape[0]
    n = data.shape[0]   #序列长度

    #三个不同的优化目标下使用交叉操作产生后代
    a = 2*int(0.15*n_2) 
    b = n_2 - 4*int(0.15*n_2)
    
    if cross_type == 'cross_op_1':
        parent_id_0 = np.random.choice(N, int(0.15*n_2),p = fitness[:,0])
        parent_id_1 = np.random.choice(N, int(0.15*n_2),p = fitness[:,0])
        target_1 = np.zeros((a,n),dtype = np.int32)
        for i in range(0,int(0.15*n_2)):
            target_1[[2*i,2*i+1],:] = cross_op_1(population[parent_id_0[i],:],population[parent_id_1[i],:])
        

        parent_id_0 = np.random.choice(N, int(0.15*n_2),p = fitness[:,1])
        parent_id_1 = np.random.choice(N, int(0.15*n_2),p = fitness[:,1])
        target_2 = np.zeros((a,n),dtype = np.int32)
        for i in range(0,int(0.15*n_2)):
            target_2[[2*i,2*i+1],:] = cross_op_1(population[parent_id_0[i],:],population[parent_id_1[i],:])

        parent_id_0 = np.random.choice(N, int(b/2),p = fitness[:,3])
        parent_id_1 = np.random.choice(N, int(b/2),p = fitness[:,3])
        target_3 = np.zeros((b,n),dtype = np.int32)
        for i in range(0,int(b/2)):
            target_3[[2*i,2*i+1],:] = cross_op_1(population[parent_id_0[i],:],population[parent_id_1[i],:])
    
    elif cross_type == 'cross_op_2':
        parent_id_0 = np.random.choice(N, int(0.15*n_2),p = fitness[:,0])
        parent_id_1 = np.random.choice(N, int(0.15*n_2),p = fitness[:,0])
        target_1 = np.zeros((a,n),dtype = np.int32)
        for i in range(0,int(0.15*n_2)):
            target_1[[2*i,2*i+1],:] = cross_op_2(population[parent_id_0[i],:],population[parent_id_1[i],:])
        

        parent_id_0 = np.random.choice(N, int(0.15*n_2),p = fitness[:,1])
        parent_id_1 = np.random.choice(N, int(0.15*n_2),p = fitness[:,1])
        target_2 = np.zeros((a,n),dtype = np.int32)
        for i in range(0,int(0.15*n_2)):
            target_2[[2*i,2*i+1],:] = cross_op_2(population[parent_id_0[i],:],population[parent_id_1[i],:])

        parent_id_0 = np.random.choice(N, int(b/2),p = fitness[:,3])
        parent_id_1 = np.random.choice(N, int(b/2),p = fitness[:,3])
        target_3 = np.zeros((b,n),dtype = np.int32)
        for i in range(0,int(b/2)):
            target_3[[2*i,2*i+1],:] = cross_op_2(population[parent_id_0[i],:],population[parent_id_1[i],:])
        
    children = np.concatenate((target_1,target_2,target_3),axis = 0)
    return children

#------------------------------------------------------------------------
'''
#三目标模拟退火算法求解
def SA(data, N,T,alpha,maxit,exit_t,search = 'reverse'):
    # data 问题的解的数目
    # N 解的数目
    # T  初始温度
    #alpha 温度下降速率 T *= alpha
    #maxit 同一温度下最大迭代次数
    #search 搜索邻域解的方式
    #exit_t 退火终止温度
    
    population = generator(data,N)
    best_cost = []  # 记录每个温度下的最优值
    mean_cost = []  # 记录每个温度下的平均值
    while T > exit_t:
        T_best = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)
        T_cost = np.zeros((maxit,4))
        for i in range(0,maxit):
            
    
    best_cost = np.min(cost,axis = 0)
    mean_record = np.zeros((T,4),dtype = np.int32)
    best_record = np.zeros((T,4),dtype = np.int32)
''''''
