import numpy as np
import pandas as pd
import loss_constraint as lc

#初始粒子群体生成 粒子位置 [-1,1]
def generator(N,L,ub,lb):
    # N 群体数   L  问题规模
    # lb left bound ub right bound
    population = (ub-lb)*np.random.rand(N,L) + lb
    #population = np.random.randn(N,L)
    return population

#
def PSO(data,N,r0,r1,T,alpha,c1,c2,c,SA_T,K,anneal,lb_x,ub_x,lb_v,ub_v):
    # N 粒子数目  T  粒子运行次数
    # r0 color_loss 正则参数 >= 0
    # r1 welding_loss 正则参数 >=0
    #lb ub 粒子及速度的边界
    # alpha 速度更新参数 0<alpha<1
    # c local search参数 0<c<1
    # c1 c2 速度更新参数  > 0 
    # SA_T 模拟退火初始温度
    # K 模拟退火搜索范围
    # anneal 模拟退火温度降低参数 0<anneal<1
    L = data.shape[0]
    population = generator(N,L,lb_x,ub_x)
    speed = generator(N,L,lb_v,ub_v)

    #population = generator(N,L)
    #speed = generator(N,L)
    #  生成生产计划     【0.1,0.4,0.2,0.7】 ->  【0,2,1,3】
    temp = np.argsort(population,axis = 1)
    permutation = np.argsort(temp,axis = 1)

    cost = np.apply_along_axis(lambda x: lc.loss(x,data,r0,r1),axis=1,arr = permutation)[:,3]


    # init p_best,g_best 
    g_best_id = np.argmin(cost)
    g_best = population[g_best_id,:]
    g_best_cost = cost[g_best_id]

    p_best = population
    p_best_cost = cost

    g_best_record = [g_best_cost] 

    for i in range(0,T):
        
        temp_1 = np.broadcast_to(np.random.rand(N,1),(N,L))
        temp_2 = np.broadcast_to(np.random.rand(N,1),(N,L))
        speed = alpha*speed + c1*temp_1*(p_best - population) + c2*temp_2*(g_best - population)
        population = population + speed
        population = np.maximum(np.minimum(population,ub_x),lb_x)
        temp = np.argsort(population,axis = 1)
        permutation = np.argsort(temp,axis = 1)

        cost = np.apply_along_axis(lambda x: lc.loss(x,data,r0,r1),axis=1,arr = permutation)[:,3]
        cur_best_id = np.argmin(cost,axis = 0)
        cur_best_cost = cost[cur_best_id]
        cur_best = population[cur_best_id,:]

        if cur_best_cost < g_best_cost:
            g_best_cost = cur_best_cost
            g_best = cur_best

        update_id = list(cost < p_best_cost)
        p_best_cost[update_id] = cost[update_id]
        p_best[update_id,:] = population[update_id,:]

        #local search
        temp = np.argsort(cost)
        rank = np.argsort(temp)
        fitness = np.power(c,rank)
        fitness = fitness/np.sum(fitness,axis = 0)
        search_id = list(fitness > np.random.randn(N))
        search_population = population[search_id,:]

        # search  SA update
        if search_population.shape[0] > 2:
            for j in range(0,K):
                search_population = np.apply_along_axis(lambda x: local_search(x,SA_T,data,r0,r1),axis=1,arr = search_population)
        else:
            for j in range(0,K):
                search_population = local_search(search_population,SA_T,data,r0,r1)
        population[search_id,:] = search_population
        SA_T = anneal*SA_T

        # update new population g_best, p_best
        temp = np.argsort(population,axis = 1)
        permutation = np.argsort(temp,axis = 1)

        cost = np.apply_along_axis(lambda x: lc.loss(x,data,r0,r1),axis=1,arr = permutation)[:,3]
        cur_best_id = np.argmin(cost)
        cur_best_cost = cost[cur_best_id]
        cur_best = population[cur_best_id,:]

        if cur_best_cost < g_best_cost:
            g_best_cost = cur_best_cost
            g_best = cur_best

        update_id = list(cost < p_best_cost)
        p_best_cost[update_id] = cost[update_id]
        p_best[update_id,:] = population[update_id,:]

        g_best_record.append(g_best_cost)

        #if i%50 == 0:
        #   print(str(i) + '  iteration' + '     best loss' + str(g_best_cost))
        print(str(i) + '  iteration' + '     best loss' + str(g_best_cost))
    return p_best,g_best_record




def local_search(x,SA_T,data,r0,r1):
# x random encode solution
    y = auto_mutation(x)

    temp = np.argsort(x)
    x_permutation = np.argsort(temp)

    temp = np.argsort(y)
    y_permutation = np.argsort(temp)

    past_cost = lc.loss(x_permutation,data,r0,r1)[3]
    cur_cost = lc.loss(y_permutation,data,r0,r1)[3]
    if cur_cost < past_cost:
        res = y
    else:
        if np.random.rand(1) < np.exp((past_cost - cur_cost)/SA_T):
            res = y
        else:
            res = x
    return res
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
    return child
